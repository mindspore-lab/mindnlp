# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore Whisper model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from ...activations import ACT2FN
from ...generation.logits_process import WhisperTimeStampLogitsProcessor
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"


WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # See all Whisper models at https://hf-mirror.com/models?filter=whisper
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function takes an attention_mask as input and performs the following operations:
    
    1. Calculates the sum of attention_mask along the last axis, treating it as a tensor of shape
    (batch_size, sequence_length).
    2. Flattens the attention_mask tensor and finds the indices of non-zero elements.
    3. Computes the maximum sequence length in the batch.
    4. Computes the cumulative sum of sequence lengths along the batch axis, padded with a zero at the beginning.

    Args:
        attention_mask (Tensor): A tensor of shape (batch_size, sequence_length) representing the attention mask.
            It is used to determine the valid elements in the input sequence.

    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A flattened tensor containing the indices of non-zero elements in the
            attention_mask tensor.
            - cu_seqlens (Tensor): A tensor of shape (batch_size + 1, sequence_length) representing the cumulative
            sum of sequence lengths, padded with a zero at the beginning.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None.
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = mindspore.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> mindspore.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = ops.exp(-log_timescale_increment * ops.arange(channels // 2))
    scaled_time = ops.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return ops.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


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


def _median_filter(inputs: mindspore.Tensor, filter_width: int) -> mindspore.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = ops.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def _dynamic_time_warping(matrix: np.ndarray):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise RuntimeError(
                f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report."
            )

    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices


class WhisperPositionalEmbedding(nn.Embedding):

    """
    Represents a Positional Embedding layer tailored for Whisper models.

    This class provides a custom implementation of positional embedding for Whisper models, inheriting from nn.Embedding.
    It allows for flexible initialization with the specified number of positions and embedding dimensions, with optional
    padding index support.

    Attributes:
        num_positions (int): The total number of positions to be embedded.
        embedding_dim (int): The dimensionality of the embedding vectors.
        padding_idx (Optional[int]): The index used for padding, if specified.

    Methods:
        __init__:
            Initializes the WhisperPositionalEmbedding instance with the given parameters.

        forward:
            Constructs the positional embeddings for the input_ids, considering past key values length when applicable.

    Returns:
        The positional embeddings corresponding to the input_ids with respect to the past key values length.

    """
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Initializes a WhisperPositionalEmbedding instance.

        Args:
            self: The instance of the class.
            num_positions (int): The number of positions to be embedded.
            embedding_dim (int): The dimension of the embedding.
            padding_idx (Optional[int]): The index used for padding sequences. Defaults to None.

        Returns:
            None.

        Raises:
            TypeError: If num_positions or embedding_dim are not integers.
            ValueError: If num_positions or embedding_dim are less than or equal to 0.
        """
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        """
        Constructs the positional embeddings for the input_ids.

        Args:
            self (WhisperPositionalEmbedding): The instance of the WhisperPositionalEmbedding class.

            input_ids (torch.Tensor): The input tensor containing the token ids. It is expected to have a shape of
                (batch_size, sequence_length).

            past_key_values_length (int, optional): The length of the past key values. Defaults to 0. This parameter is
                used to slice the positional embeddings based on the past key values length.

        Returns:
            None.

        Raises:
            None.
        """
        return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[WhisperConfig] = None,
    ):
        """
        Initializes the WhisperAttention class.

        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Default is 0.0.
            is_decoder (bool, optional): Indicates whether the attention mechanism is used as a decoder.
                Default is False.
            bias (bool, optional): Indicates whether the linear layers have bias terms. Default is True.
            is_causal (bool, optional): Indicates whether the attention is causal. Default is False.
            config (Optional[WhisperConfig], optional): The configuration for WhisperAttention. Default is None.

        Returns:
            None.

        Raises:
            ValueError: If the embed_dim is not divisible by num_heads.
        """
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

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor for attention computation.

        Args:
            self (WhisperAttention): An instance of the WhisperAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
                It should have shape (bsz * seq_len, self.embed_dim).
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    # Copied from transformers.models.bart.modeling_bart.BartAttention.forward with BART->whisper
    def forward(
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
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
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

        attn_weights = ops.softmax(attn_weights, dim=-1)

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

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

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


WHISPER_ATTENTION_CLASSES = {
    "default": WhisperAttention,
}


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):

    """
    The `WhisperEncoderLayer` class represents a single layer of the Whisper Encoder, which is used in the training
    and inference process of the Whisper model. This class inherits from the `nn.Module` class.

    Attributes:
        `embed_dim` (int): The dimension size of the input embedding.
        `self_attn` (nn.Module): The self-attention module used in the encoder layer.
        `self_attn_layer_norm` (nn.LayerNorm): Layer normalization module applied to the output of the
            self-attention module.
        `dropout` (float): Dropout probability applied to the output of the self-attention module.
        `activation_fn` (function): Activation function applied to the output of the feed-forward network.
        `activation_dropout` (float): Dropout probability applied to the output of the activation function.
        `fc1` (nn.Linear): First fully connected layer of the feed-forward network.
        `fc2` (nn.Linear): Second fully connected layer of the feed-forward network.
        `final_layer_norm` (nn.LayerNorm): Layer normalization module applied to the output of the feed-forward network.

    Methods:
        `forward`: Constructs the encoder layer by applying the self-attention, feed-forward network, and residual
            connections to the input hidden states.

    Args:
        `hidden_states` (mindspore.Tensor): The input to the layer of shape `(batch, seq_len, embed_dim)`.
        `attention_mask` (mindspore.Tensor): The attention mask of size `(batch, 1, tgt_len, src_len)`,
            where padding elements are indicated by very large negative values.
        `layer_head_mask` (mindspore.Tensor): The mask for attention heads in a given layer of size
            `(encoder_attention_heads,)`.
        `output_attentions` (bool, optional): Whether or not to return the attentions tensors of all attention layers.

    Returns:
        `(mindspore.Tensor)`: The output hidden states of the encoder layer.

    Raises:
        None

    Note:
        The forward method does not include the signatures or any other code.
    """
    def __init__(self, config: WhisperConfig):
        """
        Initializes a new instance of the WhisperEncoderLayer class.

        Args:
            self: The instance of the class.
            config (WhisperConfig): The configuration object for the WhisperEncoderLayer.
                It contains various settings and parameters for the WhisperEncoderLayer.

                - config.d_model (int): The embedding dimension.
                - config._flash_attn_2_enabled (bool, optional): Whether to enable the flash_attention_2.
                Defaults to False.
                - config.encoder_attention_heads (int): The number of attention heads in the self-attention layer.
                - config.attention_dropout (float): The dropout probability for the attention layer.
                - config.dropout (float): The dropout probability for the layer.
                - config.activation_function (str): The activation function to be used.
                - config.activation_dropout (float): The dropout probability for the activation function.
                - config.encoder_ffn_dim (int): The dimension of the feed-forward network.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.d_model
        attn_type = "flash_attention_2" if getattr(config, "_flash_attn_2_enabled", False) else "default"

        self.self_attn = WHISPER_ATTENTION_CLASSES[attn_type](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_head_mask: mindspore.Tensor,
        output_attentions: bool = False,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == mindspore.float16 and (
            ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperDecoderLayer(nn.Module):

    """
    The WhisperDecoderLayer class represents a single layer of the Whisper decoder model, which includes self-attention
    and cross-attention mechanisms. This class is designed to be used within the WhisperTransformer model for
    sequence-to-sequence tasks.

    This class inherits from nn.Module and contains methods for initializing the layer and performing computations on
    input tensors. The layer consists of self-attention, encoder attention, feedforward neural network, and layer
    normalization modules.

    The __init__ method sets up the layer with parameters such as embedding dimensions, attention types, dropout rates,
    activation functions, and normalization layers.

    The forward method processes input hidden states through the self-attention mechanism, followed by encoder
    attention if provided. It also handles dropout, residual connections, and feedforward network transformations.
    The method allows for caching of key-value states and optionally returns attention weights and cached states.

    Please refer to the method docstrings for detailed information on the input and output parameters, as well as
    their respective shapes and purposes.
    """
    def __init__(self, config: WhisperConfig):
        """
        Initializes a WhisperDecoderLayer object.

        Args:
            self (WhisperDecoderLayer): The current instance of the WhisperDecoderLayer class.
            config (WhisperConfig): An instance of the WhisperConfig class containing configuration settings.

        Returns:
            None.

        Raises:
            ValueError: If the attention type specified in the config is not supported.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__()
        self.embed_dim = config.d_model
        attn_type = "default"

        self.self_attn = WHISPER_ATTENTION_CLASSES[attn_type](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.encoder_attn = WHISPER_ATTENTION_CLASSES[attn_type](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`mindspore.Tensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(mindspore.Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class WhisperPreTrainedModel(PreTrainedModel):

    """
    This class represents a pre-trained model for the Whisper framework. It inherits from the PreTrainedModel class,
    providing additional functionality and customization specific to Whisper models.

    The class contains methods for initializing weights and for computing the output length of convolutional layers.
    The _init_weights method initializes the weights of various types of neural network cells, including dense,
    convolutional, and embedding layers, as well as custom WhisperEncoder cells. The _get_feat_extract_output_lengths
    method computes the output length of convolutional layers based on the input lengths provided.

    Overall, the WhisperPreTrainedModel class serves as a foundational framework for creating and customizing
    pre-trained models within the Whisper environment, offering flexibility in weight initialization and feature
    extraction output length computations.
    """
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, cell):
        """
        Initializes weights for the specified cell based on the configuration settings of the WhisperPreTrainedModel.

        Args:
            self (WhisperPreTrainedModel): The instance of the WhisperPreTrainedModel class.
            cell (nn.Module): The neural network cell for which weights are to be initialized.
                It can be an instance of nn.Linear, nn.Conv1d, nn.Embedding, or WhisperEncoder.

        Returns:
            None.

        Raises:
            TypeError: If the cell parameter is not an instance of nn.Module.
            ValueError: If the cell parameter is not one of the supported types
                (nn.Linear, nn.Conv1d, nn.Embedding, or WhisperEncoder).
            ValueError: If the cell type is nn.Embedding and the padding index is not provided.
            ValueError: If the cell type is WhisperEncoder and the embed_positions weight shape is not
                compatible with the sinusoids function output.
        """
        std = self.config.init_std
        if isinstance(cell, (nn.Linear, nn.Conv1d)):
            cell.weight.set_data(initializer(Normal(std),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, WhisperEncoder):
            embed_positions = cell.embed_positions.weight
            embed_positions.set_data(sinusoids(*embed_positions.shape))

    def _get_feat_extract_output_lengths(self, input_lengths: mindspore.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """
    def __init__(self, config: WhisperConfig):
        """Initialize a WhisperEncoder object.

        Args:
            config (WhisperConfig):
                The configuration object containing the parameters for the encoder.

                - dropout (float): The dropout probability for the encoder.
                - encoder_layerdrop (float): The probability of dropping an entire encoder layer.
                - d_model (int): The embedding dimension size.
                - num_mel_bins (int): The number of mel bins for the input audio.
                - pad_token_id (int): The padding token ID.
                - max_source_positions (int): The maximum number of source positions.
                - scale_embedding (bool): Whether to scale the embeddings by math.sqrt(embed_dim).

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=True)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.weight.requires_grad = False

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm([config.d_model])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        """
        Freeze the parameters of the WhisperEncoder.

        Args:
            self (WhisperEncoder): The instance of WhisperEncoder.

        Returns:
            None.

        Raises:
            None.
        """
        for param in self.get_parameters():
            param.requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        """
        Get the input embeddings for the WhisperEncoder.

        Args:
            self (WhisperEncoder): The instance of the WhisperEncoder class.

        Returns:
            nn.Module: The input embeddings.

        Raises:
            None.
        """
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        """
        Method to set input embeddings for the WhisperEncoder class.

        Args:
            self (WhisperEncoder): The instance of the WhisperEncoder class.
                It is used to access the attributes and methods of the class.
            value (nn.Module): The input embeddings to be set for the WhisperEncoder.
                It should be an instance of the nn.Module class.

        Returns:
            None: This method sets the input embeddings for the WhisperEncoder instance.

        Raises:
            None.
        """
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`mindspore.Tensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `mindspore.Tensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`mindspore.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`mindspore.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
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
        inputs_embeds = ops.gelu(self.conv1(input_features))
        inputs_embeds = ops.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.shape[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = ops.rand((1,))
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """
    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        """
        Initializes the WhisperDecoder class.

        Args:
            self: The instance of the class.
            config (WhisperConfig):
                An instance of WhisperConfig containing the configuration parameters for the decoder.

                - dropout (float): The dropout probability.
                - decoder_layerdrop (float): The layer dropout probability for the decoder.
                - pad_token_id (int): The token id used for padding.
                - max_target_positions (int): The maximum target sequence length.
                - max_source_positions (int): The maximum source sequence length.
                - d_model (int): The dimensionality of the model.
                - scale_embedding (bool): Indicates whether to scale the embeddings.
                - vocab_size (int): The size of the vocabulary.
                - decoder_layers (int): The number of decoder layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm([config.d_model])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the WhisperDecoder class.

        Args:
            self: An instance of the WhisperDecoder class.

        Returns:
            embed_tokens: This method returns the input embeddings for the decoder.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the WhisperDecoder class.

        Args:
            self (WhisperDecoder): The instance of the WhisperDecoder class.
            value: The input embeddings to be set for the WhisperDecoder.
                This parameter should be of the appropriate type and format required for input embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.shape[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand((1,))
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class WhisperModel(WhisperPreTrainedModel):

    """
    WhisperModel
    Represents a Whisper model for sequence-to-sequence tasks.

    This class inherits from WhisperPreTrainedModel and provides methods for initializing the model,
    accessing input embeddings, accessing the encoder and decoder, freezing the encoder, masking input features,
    and forwarding the model with various input parameters.

    Example:
        ```python
        >>> from transformers import AutoFeatureExtractor, WhisperModel
        >>> from datasets import load_dataset
        ...
        >>> model = WhisperModel.from_pretrained("openai/whisper-base")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 512]
        ```
    """
    def __init__(self, config: WhisperConfig):
        """
        Initializes an instance of the WhisperModel class.

        Args:
            self: The instance of the class.
            config (WhisperConfig): The configuration object used for initialization.
                This object contains various settings and parameters required for the model.
                It should be an instance of the WhisperConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the WhisperModel.

        Args:
            self (WhisperModel): The instance of the WhisperModel class.

        Returns:
            None: This method returns the input embeddings for the WhisperModel.

        Raises:
            None
        """
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the WhisperModel.

        Args:
            self (WhisperModel): The instance of the WhisperModel class.
            value (object): The input embeddings to be set for the decoder embed_tokens.

        Returns:
            None.

        Raises:
            None.
        """
        self.decoder.embed_tokens = value

    def get_encoder(self):
        """
        This method returns the encoder associated with the WhisperModel.

        Args:
            self (WhisperModel): The instance of the WhisperModel class.

        Returns:
            encoder: This method returns the encoder associated with the WhisperModel.

        Raises:
            None
        """
        return self.encoder

    def get_decoder(self):
        """
        Retrieve the decoder used in the WhisperModel.

        Args:
            self (WhisperModel): The instance of the class.

        Returns:
            decoder: This method does not return any value.

        Raises:
            None.
        """
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.shape

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = mindspore.tensor(mask_time_indices, dtype=mindspore.bool_)
            mask_time_indices = mask_time_indices[:, None].broadcast_to((-1, hidden_size, -1))
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = mindspore.tensor(mask_feature_indices, dtype=mindspore.bool_)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(
        self,
        input_features: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        r"""

        Returns:
            `Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]`

        Example:
            ```python
            >>> from transformers import AutoFeatureExtractor, WhisperModel
            >>> from datasets import load_dataset
            ...
            >>> model = WhisperModel.from_pretrained("openai/whisper-base")
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
            >>> input_features = inputs.input_features
            >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
            >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
            >>> list(last_hidden_state.shape)
            [1, 2, 512]
            ```
         """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class WhisperForConditionalGeneration(WhisperPreTrainedModel):

    """
    The `WhisperForConditionalGeneration` class is a model class for conditional text generation, inheriting from 
    `WhisperPreTrainedModel`. It provides methods for initializing the model, generating sequences of token ids, 
    preparing inputs for generation, and extracting token-level timestamps for predicted tokens.

    The class contains methods such as `forward`, `generate`, `prepare_inputs_for_generation`, and `_reorder_cache` 
    for handling conditional generation tasks. It also includes methods for freezing the encoder, getting the encoder 
    and decoder, and managing the input and output embeddings.

    The class's main methods include:
    
    - `forward`: Prepares inputs and generates sequences of token ids for conditional text generation, allowing for 
    the configuration of various generation parameters.
    - `generate`: Generates sequences of token ids for models with a language modeling head, allowing for custom logits 
    processors, stopping criteria, and other advanced generation parameters.
    - `prepare_inputs_for_generation`: Prepares input data for generation, including decoder input ids, past key values, 
    cache usage, encoder outputs, and attention masks.
    - `_reorder_cache`: Reorders the past key values based on beam indices during generation.
    - `_extract_token_timestamps`: Calculates token-level timestamps using encoder-decoder cross-attentions and dynamic 
    time-warping (DTW) to map each output token to a position in the input audio.

    This class provides a comprehensive set of tools for conditional text generation tasks, including multilingual 
    and multitask generation support, as well as token-level timestamps extraction for predicted tokens.

    For more details on how to use the class and its methods, including code examples, refer to the official 
    documentation and the [following guide](./generation_strategies).
    """
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        """
        Initializes an instance of the WhisperForConditionalGeneration class.

        Args:
            self (WhisperForConditionalGeneration): The instance of the class itself.
            config (WhisperConfig): An instance of WhisperConfig containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            AssertionError: If the config parameter is not of type WhisperConfig.
            ValueError: If an unexpected error occurs during initialization.
        """
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Retrieves the encoder from the model instance.

        Args:
            self (WhisperForConditionalGeneration): The object instance.

        Returns:
            None.

        Raises:
            None.

        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        This method 'get_decoder' is part of the class 'WhisperForConditionalGeneration' and retrieves 
        the decoder from the model.

        Args:
            self:
                Instance of the 'WhisperForConditionalGeneration' class.

                - Type: object
                - Purpose: Represents the current instance of the class.
                - Restrictions: This parameter is required for accessing the decoder.

        Returns:
            None:

                - Type: None
                - Purpose: The method returns None as it retrieves the decoder from the model.

        Raises:
            None.
        """
        return self.model.get_decoder()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the WhisperForConditionalGeneration class.

        Args:
            self:
                An instance of the WhisperForConditionalGeneration class.

                - Type: WhisperForConditionalGeneration
                - Purpose: Represents the current object of the class.
                - Restrictions: Must be an instance of WhisperForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the WhisperForConditionalGeneration class.

        Args:
            self (WhisperForConditionalGeneration): The instance of the WhisperForConditionalGeneration class.
            new_embeddings (any): The new embeddings to be set as the output embeddings for the 
                WhisperForConditionalGeneration class. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the input embeddings for the WhisperForConditionalGeneration model.

        Args:
            self (WhisperForConditionalGeneration): The instance of the WhisperForConditionalGeneration class.

        Returns:
            nn.Module: The input embeddings for the model.

        Raises:
            None.
        """
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[mindspore.Tensor]] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
                or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
                only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]

        Example:
            ```python
            >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
            >>> from datasets import load_dataset
            ...
            >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
            >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
            ...
            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            ...
            >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
            >>> input_features = inputs.input_features
            ...
            >>> generated_ids = model.generate(inputs=input_features)
            ...
            >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            >>> transcription
            ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss = ops.cross(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(
        self,
        inputs: Optional[mindspore.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: Optional[mindspore.Tensor] = None,
        return_token_timestamps=None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`mindspore.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, mindspore.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`mindspore.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`:
                A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
                `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.
                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )

            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False

        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

        forced_decoder_ids = None

        # Legacy code for backward compatibility
        if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids

        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= self.config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {self.config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _rank, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

        if generation_config.return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

        if return_token_timestamps:
            kwargs["output_attentions"] = True
            kwargs["return_dict_in_generate"] = True

            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            if kwargs.get("num_frames") is not None:
                generation_config.num_frames = kwargs.pop("num_frames")

        outputs = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            **kwargs,
        )

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            outputs["token_timestamps"] = self._extract_token_timestamps(
                outputs, generation_config.alignment_heads, num_frames=num_frames
            )

        return outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        Args:
            self (WhisperForConditionalGeneration): The instance of the WhisperForConditionalGeneration class.
            decoder_input_ids (torch.Tensor): The input tensor for the decoder.
                Shape: (batch_size, sequence_length).
            past_key_values (tuple, optional): The past key values for caching computations in auto-regressive decoding.
                Default: None.
            use_cache (bool, optional): Whether to use caching for fast decoding.
                Default: None.
            encoder_outputs (torch.Tensor, optional): The output of the encoder.
                Shape: (batch_size, sequence_length, hidden_size).
                Default: None.
            attention_mask (torch.Tensor, optional): The attention mask for the decoder input.
                Shape: (batch_size, sequence_length).
                Default: None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared inputs for generation.
                It includes the following keys:

                - 'encoder_outputs' (torch.Tensor): The output of the encoder.
                - 'past_key_values' (tuple): The past key values for caching computations in auto-regressive decoding.
                - 'decoder_input_ids' (torch.Tensor): The input tensor for the decoder.
                - 'use_cache' (bool): Whether to use caching for fast decoding.
                - 'decoder_attention_mask' (None): The attention mask for the decoder input.

        Raises:
            None.
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache according to the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
                Each element in the tuple is a tensor representing the past state of a layer.
            beam_idx (Tensor): A tensor containing the indices of the beams to be reordered.

        Returns:
            tuple: A tuple containing the reordered past key-value states for each layer. Each element in the tuple
                is a tensor representing the reordered past state of a layer.

        Raises:
            None.

        This static method takes the past key-value states and a beam index tensor, and reorders the past key-value
            states according to the beam index. It returns the reordered past key-value states as a
            tuple, where each element in the tuple represents the reordered past state of a layer.

        Note:
            The returned reordered_past tuple has the same length as the number of layers in the model, and each
                element in the tuple has the same shape as the corresponding element in the past_key_values tuple.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        # Create a list with `decoder_layers` elements, each a tensor of shape
        # (batch size, attention_heads, output length, input length).
        cross_attentions = []
        for i in range(self.config.decoder_layers):
            cross_attentions.append(ops.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = ops.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        weights = weights.permute([1, 0, 2, 3])
        if num_frames is not None:
            weights = weights[..., : num_frames // 2]

        # Normalize and smoothen the weights.
        std, mean = ops.std_mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / std
        weights = _median_filter(weights, self.config.median_filter_width)

        # Average the different cross-attention heads.
        matrix = weights.mean(axis=1)

        timestamps = ops.zeros_like(generate_outputs.sequences, dtype=mindspore.float32)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(timestamps.shape[0]):
            text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].asnumpy())
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = mindspore.tensor(jump_times)

        return timestamps


class WhisperDecoderWrapper(WhisperPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the WhisperDecoderWrapper class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing the settings for the decoder.
                The config object should have the following attributes:

                - is_encoder_decoder (bool): Indicates if the WhisperDecoderWrapper is used as an encoder-decoder.
                This should be set to False for the WhisperDecoderWrapper class.
                - Other attributes specific to the WhisperDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        config.is_encoder_decoder = False
        self.decoder = WhisperDecoder(config)

    def get_input_embeddings(self):
        """
        Get input embeddings for the WhisperDecoderWrapper.

        Args:
            self (WhisperDecoderWrapper):
                The instance of WhisperDecoderWrapper for which input embeddings are to be retrieved.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the WhisperDecoderWrapper.

        Args:
            self (WhisperDecoderWrapper): The instance of the WhisperDecoderWrapper class.
            value (object): The input embeddings to be set for the decoder. It should be an object of the
                desired input embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.decoder.embed_tokens = value

    def forward(self, *args, **kwargs):
        """
        Method to forward a WhisperDecoderWrapper object by invoking the decoder with the provided arguments.

        Args:
            self (WhisperDecoderWrapper): The instance of the WhisperDecoderWrapper class.

        Returns:
            None: This method does not return any value explicitly. It delegates the forwardion to the decoder method.

        Raises:
            None.
        """
        return self.decoder(*args, **kwargs)


class WhisperForCausalLM(WhisperPreTrainedModel):

    """
    WhisperForCausalLM is a class representing a Whisper model for causal language modeling tasks.
    It inherits from WhisperPreTrainedModel and provides methods for generating text based on input sequences.

    Methods:
        __init__: Initializes the WhisperForCausalLM model with the given configuration.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets new output embeddings for the model.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets new input embeddings for the model.
        set_decoder: Sets the decoder for the model.
        get_decoder: Returns the decoder of the model.
        forward: Constructs the model architecture and processes input data for generation.
        prepare_inputs_for_generation: Prepares inputs for text generation based on the provided parameters.
        _reorder_cache: Reorders cache items based on a given beam index for generation.

    Example:
        ```python
        >>> from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
        >>> from datasets import load_dataset
        ...
        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        ...
        >>> assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")
        ...
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> sample = ds[0]["audio"]
        >>> input_features = processor(
        ...     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ... ).input_features
        ...
        >>> predicted_ids = model.generate(input_features, assistant_model=assistant_model)
        ...
        >>> # Decode token ids to text
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'
        ```
    """
    _tied_weights_keys = ["proj_out.weight"]
    main_input_name = "input_ids"

    def __init__(self, config):
        """
        Initializes an instance of the WhisperForCausalLM class.

        Args:
            self (WhisperForCausalLM): The instance of the class.
            config: A configuration object containing various settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        config.is_encoder_decoder = False
        self.model = WhisperDecoderWrapper(config)

        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for WhisperForCausalLM model.

        Args:
            self: The instance of WhisperForCausalLM class.

        Returns:
            proj_out: This method returns the output embeddings.

        Raises:
            None.
        """
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the WhisperForCausalLM model.

        Args:
            self (WhisperForCausalLM): The instance of WhisperForCausalLM class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of the correct type.
            ValueError: If any restrictions or validations fail during the setting of new embeddings.
        """
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        """
        Retrieves the input embeddings from the underlying model.

        Args:
            self (WhisperForCausalLM): The instance of the WhisperForCausalLM class.

        Returns:
            nn.Module: The input embeddings obtained from the underlying model.

        Raises:
            None.

        Description:
            This method returns the input embeddings of the WhisperForCausalLM model.
            The input embeddings are responsible for mapping the input tokens to their corresponding embedding vectors.
            The underlying model's 'get_input_embeddings' function is called to retrieve these embeddings.

        Note:
            - The returned input embeddings can be used for various downstream tasks such as fine-tuning or feature
            extraction.
            - It is assumed that the underlying model has a 'get_input_embeddings' method implemented.

        Example:
            ```python
            >>> model = WhisperForCausalLM()
            >>> embeddings = model.get_input_embeddings()
            ```
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the WhisperForCausalLM model.

        Args:
            self: The object instance.
            value: A tensor of shape (vocab_size, hidden_size) representing the new input embeddings for the model.
                The vocab_size is the size of the vocabulary used by the model, and the hidden_size is the size of
                the hidden states in the model. The input embeddings are used to encode the input tokens in the model's
                forward pass. This parameter is required.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        """
        Method to set the decoder for the WhisperForCausalLM model.

        Args:
            self (WhisperForCausalLM): The instance of the WhisperForCausalLM class.
            decoder: The decoder to be set for the model. It should be compatible with the model's decoder architecture.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Returns the decoder of the WhisperForCausalLM model.

        Args:
            self: The instance of the WhisperForCausalLM class.

        Returns:
            decoder: This method returns the decoder of the WhisperForCausalLM model.

        Raises:
            None.
        """
        return self.model.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_outputs  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model. Contains
                pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used (see `past_key_values` input) to speed up sequential decoding. If
                `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]

        Example:
            ```python
            >>> from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
            >>> from datasets import load_dataset
            ...
            >>> processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
            >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
            ...
            >>> assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")
            ...
            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            >>> sample = ds[0]["audio"]
            >>> input_features = processor(
            ...     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
            ... ).input_features
            ...
            >>> predicted_ids = model.generate(input_features, assistant_model=assistant_model)
            ...
            >>> # decode token ids to text
            >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            >>> transcription
            ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If the user passed a tuple or `BaseModelOutput` for encoder_outputs, we extract only the hidden states
        if isinstance(encoder_outputs, (BaseModelOutput, tuple, list)):
            encoder_outputs = encoder_outputs[0]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        Args:
            self (object): The instance of the class.
            input_ids (Tensor): The input tensor containing the token ids.
            past_key_values (tuple, optional): The past key values for efficient generation. Defaults to None.
            use_cache (bool, optional): Whether to use caching for the generation process. Defaults to None.
            encoder_outputs (Tensor, optional): The outputs of the encoder. Defaults to None.
            attention_mask (Tensor, optional): The attention mask for the input_ids. Defaults to None.

        Returns:
            dict: A dictionary containing the prepared inputs for generation including encoder_outputs, past_key_values,
                input_ids, use_cache, and attention_mask.

        Raises:
            ValueError: If the input_ids and past_key_values are not of compatible shape.
            IndexError: If the input_ids shape is not as expected.
        """
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
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "input_ids": input_ids,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache of past key values for the beam search in the WhisperForCausalLM class.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each element in the tuple is expected to be a tensor.
            beam_idx (tensor): The indices of the beams for reordering the past key values.

        Returns:
            None: This method modifies the past_key_values in place.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class WhisperForAudioClassification(WhisperPreTrainedModel):

    """
    This class represents a Whisper model for audio classification tasks.
    It is a subclass of the `WhisperPreTrainedModel` class.

    The `WhisperForAudioClassification` class consists of various methods and attributes that are used for audio
    classification tasks.

    Methods:
        `__init__`: Initializes the `WhisperForAudioClassification` instance.
        `freeze_encoder`: Disables gradient computation for the Whisper encoder, preventing its parameters
            from being updated during training.
        `get_input_embeddings`: Retrieves the input embeddings from the encoder.
        `set_input_embeddings`: Sets the input embeddings for the encoder.
        `forward`: Constructs the forward pass of the model for audio classification.

    Attributes:
        `encoder`: Instance of the `WhisperEncoder` class used for encoding audio input.
        `layer_weights`: Parameter representing weights for weighted layer sum, if enabled.
        `projector`: Instance of the `nn.Linear` class used for projecting hidden states.
        `classifier`: Instance of the `nn.Linear` class used for classification.
        `config`: Configuration object containing model settings.

    Example:
        ```python
        >>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
        >>> from datasets import load_dataset
        ...
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        >>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        ...
        >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        >>> sample = next(iter(ds))
        ...
        >>> inputs = feature_extractor(
        ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
        ... )
        >>> input_features = inputs.input_features
        ...
        >>> with torch.no_grad():
        >>>     logits = model(input_features).logits
        ...
        >>> predicted_class_ids = torch.argmax(logits).item()
        >>> predicted_label = model.config.id2label[predicted_class_ids]
        >>> predicted_label
        'Afrikaans'
        ```

    For more details on the class methods and attributes, refer to the individual method docstrings.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the WhisperForAudioClassification class.

        Args:
            self: The instance of the WhisperForAudioClassification class.
            config: A configuration object containing settings for the model.
                It should be an instance of the configuration class specific to WhisperForAudioClassification.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is not of the expected type.
            ValueError: If the 'num_hidden_layers' attribute in the 'config' parameter is not defined.
            RuntimeError: If an error occurs during initialization.
        """
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = Parameter(ops.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training. Only the projection layers and classification head will be updated.
        """
        self.encoder._freeze_parameters()

    def get_input_embeddings(self) -> nn.Module:
        """
        This method returns the input embeddings from the encoder for audio classification.

        Args:
            self (WhisperForAudioClassification): The instance of the WhisperForAudioClassification class.

        Returns:
            nn.Module: The input embeddings from the encoder for audio classification.

        Raises:
            None
        """
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        """
        Method to set the input embeddings for the WhisperForAudioClassification class.

        Args:
            self:
                The instance of the WhisperForAudioClassification class.

                - Type: WhisperForAudioClassification
                - Purpose: Represents the current instance of the class.
                - Restrictions: None

            value:
                The input embeddings to be set for the encoder.

                - Type: nn.Module
                - Purpose: Represents the input embeddings used for encoding.
                - Restrictions: Should be an instance of nn.Module.

        Returns:
            None.

        Raises:
            None.
        """
        self.encoder.set_input_embeddings(value)

    def forward(
        self,
        input_features: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]

        Example:
            ```python
            >>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
            >>> from datasets import load_dataset
            ...
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
            >>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
            ...
            >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
            >>> sample = next(iter(ds))
            ...
            >>> inputs = feature_extractor(
            ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
            ... )
            >>> input_features = inputs.input_features
            ...
            >>> with torch.no_grad():
            ...     logits = model(input_features).logits
            ...
            >>> predicted_class_ids = torch.argmax(logits).item()
            >>> predicted_label = model.config.id2label[predicted_class_ids]
            >>> predicted_label
            'Afrikaans'
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = ops.stack(encoder_outputs, dim=1)
            norm_weights = ops.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(axis=1)

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

__all__ = [
    "WhisperForCausalLM",
    "WhisperForConditionalGeneration",
    "WhisperModel",
    "WhisperPreTrainedModel",
    "WhisperForAudioClassification",
]
