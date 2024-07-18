# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# pylint: disable=too-many-lines
""" MindSpore SeamlessM4Tv2 model."""


import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import ops, nn, Parameter
from mindspore.common.initializer import initializer, Normal, XavierUniform, Uniform, HeNormal

from mindnlp.utils import ModelOutput, logging, get_default_dtype
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config


logger = logging.get_logger(__name__)


SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/seamless-m4t-v2-large",
    # See all SeamlessM4T-v2 models at https://hf-mirror.com/models?filter=seamless_m4t_v2
]


SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://hf-mirror.com/microsoft/speecht5_hifigan/resolve/main/config.json",
}


@dataclass
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TGenerationOutput with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2GenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4Tv2Model`], [`SeamlessM4Tv2ForTextToText`],
    [`SeamlessM4Tv2ForTextToSpeech`], [`SeamlessM4Tv2ForSpeechToSpeech`] and [`SeamlessM4Tv2ForTextToSpeech`].

    Args:
        waveform (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        waveform_lengths (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        sequences (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        unit_sequences (`mindspore.Tensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """
    waveform: Optional[mindspore.Tensor] = None
    waveform_lengths: Optional[mindspore.Tensor] = None
    sequences: Optional[Tuple[mindspore.Tensor]] = None
    unit_sequences: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class SeamlessM4Tv2TextToUnitDecoderOutput(ModelOutput):
    """
    Class defining the outputs from [`SeamlessM4Tv2TextToUnitDecoder`].

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        padding_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
    """
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    padding_mask: Optional[mindspore.Tensor] = None


@dataclass
class SeamlessM4Tv2TextToUnitOutput(ModelOutput):
    """
    Class defining the outputs from [`SeamlessM4Tv2TextToUnitForConditionalGeneration`] and
    [`SeamlessM4Tv2TextToUnitModel`].

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        padding_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*
        decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True`
            is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
    """
    last_hidden_state: mindspore.Tensor = None
    padding_mask: Optional[mindspore.Tensor] = None
    decoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    decoder_attentions: Optional[Tuple[mindspore.Tensor]] = None
    encoder_last_hidden_state: Optional[mindspore.Tensor] = None
    encoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    encoder_attentions: Optional[Tuple[mindspore.Tensor]] = None
    loss: Optional[mindspore.Tensor] = None


############ UTILS ################
# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
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


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].copy()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _compute_new_attention_mask(hidden_states: mindspore.Tensor, seq_lens: mindspore.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`mindspore.Tensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`mindspore.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `mindspore.Tensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = ops.arange(mask_seq_len).broadcast_to((batch_size, -1))

    bool_mask = indices >= seq_lens.unsqueeze(1).broadcast_to((-1, mask_seq_len))

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.format_speech_generation_kwargs with SeamlessM4T->SeamlessM4Tv2
def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4Tv2 models that generate speech, attribute kwargs to either the text generation or the
    speech generation models.

    Args:
        kwargs (`dict`)`:
            Keyword arguments are of two types:

            - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
            except for `decoder_input_ids` which will only be passed through the text components.
            - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
            text model and speech model respectively. It has the priority over the keywords without a prefix.

            This means you can, for example, specify a generation strategy for one generation but not for the
            other.
    """
    # attribute kwargs to models
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith("text_"):
            key = key[len("text_") :]
            kwargs_text[key] = value
        elif key.startswith("speech_"):
            key = key[len("speech_") :]
            kwargs_speech[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return kwargs_text, kwargs_speech


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """
    Pad a list of sequences to the same length.

    Args:
        sequences (List[List[float]]): The list of sequences to be padded.
        batch_first (bool, optional): If True, the output tensor will have shape (batch_size, max_len, features).
            If False, the shape will be (max_len, batch_size, features). Default is False.
        padding_value (float, optional): The value used for padding. Default is 0.0.

    Returns:
        torch.Tensor: A tensor containing the padded sequences.

    Raises:
        None.
    """
    # Determine the maximum sequence length
    max_len = max(len(seq) for seq in sequences)

    # Pad each sequence using cp.pad
    padded_sequences = [ops.pad(seq, (0, 0, 0, max_len - len(seq)), mode='constant', value=padding_value) for seq in sequences]
    # Stack the padded sequences along the appropriate axis
    if batch_first:
        padded_sequence = ops.stack(padded_sequences)
    else:
        padded_sequence = ops.stack(padded_sequences, 1)
    return padded_sequence

############ SPEECH ENCODER related code ################

class SeamlessM4Tv2ConformerFeatureProjection(nn.Cell):

    """
    This class represents a feature projection module for the SeamlessM4Tv2Conformer model.
    It inherits from the nn.Cell class.

    The feature projection module is responsible for projecting the input hidden states into a higher-dimensional space,
    followed by layer normalization and dropout. This helps in capturing complex patterns and enhancing the expressive
    power of the model.

    Attributes:
        layer_norm (nn.LayerNorm): A layer normalization module that normalizes the hidden states.
        projection (nn.Dense): A dense linear projection layer that projects the hidden states into a
            higher-dimensional space.
        dropout (nn.Dropout): A dropout module that randomly sets elements of the hidden states to zero.

    Methods:
        __init__:
            Initializes the SeamlessM4Tv2ConformerFeatureProjection module with the given configuration.

        construct:
            Applies the feature projection operation on the input hidden states.

    Returns:
        The projected hidden states after applying layer normalization and dropout.

    Note:
        - The input hidden states should have a shape of [batch_size, sequence_length, input_dim].
        - The configuration should contain the following attributes:

            - feature_projection_input_dim: The input dimension of the feature projection layer.
            - hidden_size: The output dimension of the feature projection layer.
            - layer_norm_eps: The epsilon value for layer normalization.
            - speech_encoder_dropout: The dropout probability for the dropout layer.

    Example:
        ```python
        >>> config = {
        ...     'feature_projection_input_dim': 512,
        ...     'hidden_size': 256,
        ...     'layer_norm_eps': 1e-5,
        ...     'speech_encoder_dropout': 0.1
        ...}
        >>> feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        >>> hidden_states = torch.randn(3, 100, 512)
        >>> projected_states = feature_projection.construct(hidden_states)
        ```
    """
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeatureProjection.__init__
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4Tv2ConformerFeatureProjection class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration parameters for the feature projection.

                - feature_projection_input_dim (int): The input dimension of the feature projection.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_size (int): The size of the hidden layer.
                - speech_encoder_dropout (float): The dropout probability for the speech encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm([config.feature_projection_input_dim], epsilon=config.layer_norm_eps)
        self.projection = nn.Dense(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

    def construct(self, hidden_states):
        """Constructs the feature projection for the SeamlessM4Tv2Conformer model.

        Args:
            self (SeamlessM4Tv2ConformerFeatureProjection): An instance of the SeamlessM4Tv2ConformerFeatureProjection class.
            hidden_states (torch.Tensor): The input hidden states tensor to be projected.

        Returns:
            torch.Tensor or None: The projected hidden states tensor. If the input tensor is None, the method returns None.

        Raises:
            TypeError: If the input hidden_states tensor is not a torch.Tensor object.
            ValueError: If the input hidden_states tensor is empty or has an incompatible shape.
            RuntimeError: If the input hidden_states tensor cannot be cast to the same dtype as the layer_norm weights.
        """
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeedForward with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerFeedForward(nn.Cell):

    """
    This class represents a feed-forward module for the SeamlessM4Tv2Conformer model, which is used for speech encoding.

    Inherits from: nn.Cell

    Attributes:
        config: An object containing configuration parameters for the module.
        act_fn: The activation function to be applied to the intermediate hidden states.
        dropout: The dropout probability to be applied to the intermediate hidden states.

    Methods:
        __init__:
            Initializes the SeamlessM4Tv2ConformerFeedForward module.

            Args:

            - config: An object containing configuration parameters for the module.
            - act_fn (optional): The activation function to be applied to the intermediate hidden states.
            - dropout (optional): The dropout probability to be applied to the intermediate hidden states.

        construct:
            Applies the feed-forward operations on the input hidden states.

            Args:

            - hidden_states: The input hidden states to be processed.

            Returns:

            - hidden_states: The processed hidden states after applying the feed-forward operations.
    """
    def __init__(self, config, act_fn=None, dropout=None):
        """
        Initializes an instance of the SeamlessM4Tv2ConformerFeedForward class.

        Args:
            self: The object instance.
            config: An object containing configuration parameters.
            act_fn (optional): The activation function to be used for the hidden layers.
                If not provided, it defaults to the value of config.speech_encoder_hidden_act.
                It can be either a string specifying a predefined activation function or a custom activation function.
            dropout (optional): The dropout probability for the intermediate layers.
                If not provided, it defaults to the value of config.speech_encoder_dropout.

        Returns:
            None.

        Raises:
            None.

        Note:
            - The intermediate_dropout attribute is assigned an instance of nn.Dropout with p=dropout.
            - The intermediate_dense attribute is assigned an instance of nn.Dense with input size config.hidden_size
            and output size config.speech_encoder_intermediate_size.
            - The intermediate_act_fn attribute is assigned the activation function specified by act_fn.
            If act_fn is a string, it is mapped to the corresponding activation function from the ACT2FN dictionary.
            If act_fn is a custom function, it is directly assigned.
            - The output_dense attribute is assigned an instance of nn.Dense with input size
            config.speech_encoder_intermediate_size and output size config.hidden_size.
            - The output_dropout attribute is assigned an instance of nn.Dropout with p=dropout.
        """
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        self.intermediate_dropout = nn.Dropout(p=dropout)
        self.intermediate_dense = nn.Dense(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = ACT2FN[act_fn] if isinstance(act_fn, str) else act_fn

        self.output_dense = nn.Dense(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=dropout)

    def construct(self, hidden_states):
        """
        Constructs the feedforward layer in the SeamlessM4Tv2Conformer model.

        Args:
            self (SeamlessM4Tv2ConformerFeedForward): An instance of the SeamlessM4Tv2ConformerFeedForward class.
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None

        Description:
            This method applies a series of operations to the input hidden states to construct the feedforward layer
            in the SeamlessM4Tv2Conformer model. The operations include intermediate dense layer, activation function,
            dropout layer, and output dense layer. The resulting hidden states are returned.

            - intermediate_dense: Applies a linear transformation to the hidden states using the intermediate dense layer.
            - intermediate_act_fn: Applies the activation function to the intermediate dense outputs.
            - intermediate_dropout: Applies dropout to the intermediate outputs.
            - output_dense: Applies a linear transformation to the intermediate outputs using the output dense layer.
            - output_dropout: Applies dropout to the output dense outputs.

            Note:
                The intermediate dense layer, activation function, dropout layers, and output dense layer must be defined
                before calling this method.
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4Tv2ConformerConvolutionModule(nn.Cell):
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of `https://doi.org/10.48550/arxiv.1609.03499"""
    def __init__(self, config):
        """
        Initializes the SeamlessM4Tv2ConformerConvolutionModule.

        Args:
            self (object): The instance of the class.
            config (object):
                The configuration object containing various parameters for the module.

                - conv_depthwise_kernel_size (int): The kernel size for depthwise convolution.
                - hidden_size (int): The hidden size used in convolution layers.
                - speech_encoder_hidden_act (str): The activation function for hidden layers.
                - speech_encoder_dropout (float): The dropout rate.

        Returns:
            None.

        Raises:
            ValueError: Raised if the 'config.conv_depthwise_kernel_size' is not an odd number,
                as it should be for 'SAME' padding.
        """
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm([config.hidden_size])
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            pad_mode='valid',
            has_bias=False,
        )
        self.glu = nn.GLU(axis=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            pad_mode='valid',
            group=config.hidden_size,
            has_bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm([config.hidden_size])
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            pad_mode='valid',
            has_bias=False,
        )
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

    def construct(self, hidden_states, attention_mask=None):
        """
        Constructs the SeamlessM4Tv2ConformerConvolutionModule.

        Args:
            self: The instance of the SeamlessM4Tv2ConformerConvolutionModule.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length)
                indicating which tokens should be attended to and which should not. Defaults to None.

        Returns:
            Tensor: The output hidden states tensor after applying the convolution operations of shape
                (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.swapaxes(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = ops.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


class SeamlessM4Tv2ConformerSelfAttention(nn.Cell):
    """Construct a SeamlessM4Tv2ConformerSelfAttention object.
    Can be enhanced with relative position embeddings.
    """
    def __init__(self, config, use_position_embeddings=True):
        """
        Initializes a new instance of the SeamlessM4Tv2ConformerSelfAttention class.

        Args:
            self: The object itself.
            config: An instance of the configuration class that contains the model's configuration parameters.
            use_position_embeddings (bool): Whether to use position embeddings or not. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Dense(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Dense(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Dense(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Dense(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        if self.position_embeddings_type == "relative_key":
            self.left_max_position_embeddings = config.left_max_position_embeddings
            self.right_max_position_embeddings = config.right_max_position_embeddings
            num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
            self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the self-attention mechanism in the SeamlessM4Tv2ConformerSelfAttention class.

        Args:
            self (SeamlessM4Tv2ConformerSelfAttention): An instance of the SeamlessM4Tv2ConformerSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional attention mask tensor of shape
                (batch_size, sequence_length, sequence_length). Defaults to None.
            output_attentions (bool): Indicates whether to output the attention weights. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing:

                - attn_output (mindspore.Tensor): The attention output tensor of shape
                (batch_size, sequence_length, hidden_size).
                - attn_weights (Optional[mindspore.Tensor]): The attention weights tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length). None if output_attentions is False.
                - None (Optional[Tuple[mindspore.Tensor]]): None if output_attentions is False.

        Raises:
            None
        """
        # self-attention mechanism
        batch_size, _, _ = hidden_states.shape

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        attn_weights = ops.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(self.head_size)

        if self.position_embeddings_type == "relative_key":
            query_length, key_length = query.shape[2], key.shape[2]

            position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_r - position_ids_l
            distance = ops.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

            positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
            positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

            relative_position_attn_weights = ops.einsum("bhld,lrd->bhlr", query, positional_embedding)
            attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = ops.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SeamlessM4Tv2ConformerEncoderLayer(nn.Cell):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        """
        Initialize a SeamlessM4Tv2ConformerEncoderLayer object.

        Args:
            self (SeamlessM4Tv2ConformerEncoderLayer): The instance of the class.
            config:
                An object containing the configuration parameters for the encoder layer.

                - hidden_size (int): The dimension of the embedding.
                - speech_encoder_dropout (float): The dropout probability for the self-attention layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm([embed_dim])
        self.self_attn_dropout = nn.Dropout(p=dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm([embed_dim])

    def construct(
        self,
        hidden_states,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        Constructs a SeamlessM4Tv2ConformerEncoderLayer.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states. Shape is (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor. Default is None.
                If provided, the attention mask tensor must have the same shape as `hidden_states`.
                A value of 0 in the attention mask tensor indicates masking for the corresponding position,
                while a value of 1 indicates non-masking.
            output_attentions (bool, optional): Whether to output the attention weights. Default is False.
            conv_attention_mask (Optional[mindspore.Tensor], optional):
                The convolution attention mask tensor. Default is None.
                If provided, the convolution attention mask tensor must have the same shape as `hidden_states`.
                A value of 0 in the convolution attention mask tensor indicates masking for the corresponding position,
                while a value of 1 indicates non-masking.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor]]:
                A tuple containing:

                - hidden_states (mindspore.Tensor): The output hidden states. Shape is
                (batch_size, sequence_length, hidden_size).
                - attn_weights (Optional[mindspore.Tensor]): The attention weights tensor if
                `output_attentions` is True, else None.

        Raises:
            None.
        """
        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class SeamlessM4Tv2ConformerEncoder(nn.Cell):

    """
    The class represents a SeamlessM4Tv2ConformerEncoder, which is a neural network cell for encoding speech data.
    It inherits from the nn.Cell class.

    The class includes methods for initializing the encoder, applying chunk attention, and constructing the hidden states.
    The __init__ method initializes the encoder with the given configuration, dropout, layers, and layer normalization.
    The _apply_chunk_attention method creates a chunk attention mask to prevent attention across chunks.
    The construct method processes the hidden states, applies chunk attention if specified, and performs layer-wise
    computations.

    Note:
        This docstring is a summary based on the provided code and may need additional details from the broader context
        of the codebase.
    """
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4Tv2ConformerEncoder class.

        Args:
            self: An instance of the class.
            config:
                An object of type 'config' containing the configuration settings for the encoder.

                - Type: Config object
                - Purpose: Specifies the configuration parameters for the encoder.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)
        self.layers = nn.CellList(
            [SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        sequence_len = hidden_states.shape[1]

        chunk_indices = ops.arange(sequence_len)
        chunk_indices = ops.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        start_indices = ops.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp(max=sequence_len)

        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        indices = ops.arange(sequence_len).unsqueeze(0).expand(sequence_len, -1)

        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        attention_mask = chunk_mask if attention_mask is None else (attention_mask.bool() | chunk_mask)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Constructs the SeamlessM4Tv2ConformerEncoder.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The hidden states of the encoder. Shape should be
                (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): The attention mask tensor.
                If provided, it should have the same shape as 'hidden_states'.
                Masked positions have a value of 'True' and unmasked positions have a value of 'False'.
                Default is 'None'.
            output_attentions (bool, optional): Whether to output the self-attention tensors of each layer.
                Default is 'False'.
            output_hidden_states (bool, optional): Whether to output the hidden states of each layer. Default is 'False'.
            return_dict (bool, optional): Whether to return the output as a dictionary. Default is 'True'.

        Returns:
            None

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        if self.config.speech_encoder_chunk_size is not None:
            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask * float(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).min)

        hidden_states = self.dropout(hidden_states)

        for _, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = ops.rand([])

            skip_the_layer = bool(self.training and (dropout_probability < self.config.speech_encoder_layerdrop))
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    conv_attention_mask=conv_attention_mask,
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapterLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapterLayer(nn.Cell):
    """
    This class represents a layer for the SeamlessM4Tv2 Conformer Adapter. It inherits from nn.Cell and contains methods
    for computing sub-sample lengths from attention mask and constructing the adapter layer using the given input and
    optional attention mask.

    Attributes:
        config (object): The configuration object containing hidden size and adaptor dropout information.

    Methods:
        _compute_sub_sample_lengths_from_attention_mask(attention_mask): Computes sub-sample lengths from the
            attention mask.
        construct(hidden_states, attention_mask, output_attentions): Constructs the adapter layer using the given
            input hidden_states and optional attention_mask.

    Note:
        For detailed information on the class methods and attributes, please refer to the class code and comments.
    """
    def __init__(self, config):
        """
        This method initializes an instance of the SeamlessM4Tv2ConformerAdapterLayer class.

        Args:
            self: The instance of the class.
            config: A configuration object containing the parameters for the adapter layer.
                It is expected to have the following attributes:

                - hidden_size: An integer representing the dimension of the hidden state.
                - adaptor_dropout: A float representing the dropout probability for the adapter layer.
                - adaptor_kernel_size: An integer representing the size of the kernel for the convolutional layers
                in the adapter.
                - adaptor_stride: An integer representing the stride for the convolutional layers in the adapter.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm([embed_dim])
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            pad_mode='pad',
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(axis=1)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm([embed_dim])
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            pad_mode='pad',
            padding=self.stride // 2,
        )
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(p=dropout)

        # Feed-forward
        self.ffn_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        """
        Computes the lengths of sub-samples based on the attention mask.

        Args:
            self (SeamlessM4Tv2ConformerAdapterLayer): An instance of the SeamlessM4Tv2ConformerAdapterLayer class.
            attention_mask (Tensor): A binary tensor of shape (batch_size, sequence_length) representing the attention
                mask.

        Returns:
            None

        Raises:
            None

        This method computes the lengths of sub-samples based on the attention mask. The attention mask is a binary
        tensor where each element indicates whether the corresponding token is a valid token (1) or a padding token (0).
        The method calculates the sequence lengths for each sample in the batch by subtracting the number of padding
        tokens from the total sequence length.

        The sequence lengths are then adjusted to account for the kernel size and stride. The method applies a padding
        value 'pad' equal to half the kernel size. It subtracts twice the padding value and the kernel size from the
        sequence lengths, and then divides the result by the stride value. Finally, it adds 1 to obtain the lengths of
        the sub-samples.

        The resulting sequence lengths are converted to float32 data type using the 'astype' method and then rounded
        down to the nearest integer using the 'floor' method from the MindSpore library.

        Note:
            The returned value is of type None, as the sequence lengths are stored internally within the
            SeamlessM4Tv2ConformerAdapterLayer object.
        """
        pad = self.kernel_size // 2
        seq_lens = attention_mask.shape[1] - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.astype(mindspore.float32).floor()

    def construct(
        self,
        hidden_states,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Constructs the SeamlessM4Tv2ConformerAdapterLayer.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states. It represents the input data to the layer.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask.
                Defaults to None. If provided, it restricts the attention of the layer.
            output_attentions (bool): A flag indicating whether to output attentions. Defaults to False.

        Returns:
            mindspore.Tensor: The output hidden states after processing through the layer.

        Raises:
            ValueError: If the dimensions of input tensors are incompatible.
            RuntimeError: If an error occurs during the computation process.
            TypeError: If the input parameters are of incorrect type.
        """
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.swapaxes(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.swapaxes(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.swapaxes(1, 2)

        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapter with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapter(nn.Cell):

    """A class representing a SeamlessM4Tv2ConformerAdapter.

    Inherits from nn.Cell.

    This class initializes an instance of SeamlessM4Tv2ConformerAdapter and constructs the adapter layers.
    Each adapter layer is a SeamlessM4Tv2ConformerAdapterLayer, and the number of layers is determined by
    the 'num_adapter_layers' parameter in the configuration.

    Attributes:
        layers (nn.CellList): A list of SeamlessM4Tv2ConformerAdapterLayer instances representing the adapter layers.

    Methods:
        __init__: Initializes a new instance of SeamlessM4Tv2ConformerAdapter.
        construct: Constructs the adapter layers by iterating over each layer and applying it to the input
            hidden states and attention mask.

    """
    def __init__(self, config):
        """
        Initializes an instance of the 'SeamlessM4Tv2ConformerAdapter' class.

        Args:
            self: The instance of the 'SeamlessM4Tv2ConformerAdapter' class.
            config: An object of type 'Config' containing configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.layers = nn.CellList(
            [SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)]
        )

    def construct(self, hidden_states, attention_mask):
        """
        Constructs the hidden states of the SeamlessM4Tv2ConformerAdapter by applying the layers in sequence.

        Args:
            self (SeamlessM4Tv2ConformerAdapter): An instance of the SeamlessM4Tv2ConformerAdapter class.
            hidden_states (Tensor): The input hidden states. The shape is (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor): The attention mask tensor. The shape is (batch_size, sequence_length).

        Returns:
            None

        Raises:
            None
        """
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding
class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Cell):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Initialize the SeamlessM4Tv2SinusoidalPositionalEmbedding class.

        Args:
            self: The instance of the class.
            num_positions (int): The number of positions to be embedded.
            embedding_dim (int): The dimension of the embedding vector.
            padding_idx (Optional[int], optional): The index used for padding. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        This method initializes and assigns embedding weights to the 'weights' attribute of the
        'SeamlessM4Tv2SinusoidalPositionalEmbedding' class.

        Args:
            self: The instance of the 'SeamlessM4Tv2SinusoidalPositionalEmbedding' class.
            num_embeddings (int): The number of unique embeddings to be used.
            embedding_dim (int): The dimensionality of the embedding vector.
            padding_idx (Optional[int], optional): The index to ignore in the embeddings. Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype of the param
            emb_weights = emb_weights.to(self.weights.dtype) # pylint: disable=access-member-before-definition
        self.weights = emb_weights

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim, dtype=mindspore.float32) * -emb)
        emb = ops.arange(num_embeddings, dtype=mindspore.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = ops.cat([emb, ops.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(get_default_dtype())

    def construct(
        self, input_ids: mindspore.Tensor = None, inputs_embeds: mindspore.Tensor = None, past_key_values_length: int = 0
    ):
        """
        Constructs a sinusoidal positional embedding for the SeamlessM4Tv2SinusoidalPositionalEmbedding class.

        Args:
            self (SeamlessM4Tv2SinusoidalPositionalEmbedding):
                An instance of the SeamlessM4Tv2SinusoidalPositionalEmbedding class.
            input_ids (mindspore.Tensor, optional): The input tensor that contains the tokenized input sequence.
                Default is None.
            inputs_embeds (mindspore.Tensor, optional): The input tensor that contains the embedded input sequence.
                Default is None.
            past_key_values_length (int, optional): The length of past key values to be used in the positional
                embedding calculation. Default is 0.

        Returns:
            None.

        Raises:
            None.
        """
        if input_ids is not None:
            bsz, seq_len = input_ids.shape
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            bsz, seq_len = inputs_embeds.shape[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.shape[0]:
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1])

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: mindspore.Tensor

        Returns: mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64)
        return position_ids.unsqueeze(0).broadcast_to(input_shape) + past_key_values_length


class SeamlessM4Tv2Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Copied from transformers.models.bart.modeling_bart.BartAttention.__init__ with Bart->SeamlessM4Tv2
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4Tv2Config] = None,
    ):
        """Initializes the SeamlessM4Tv2Attention object.

        Args:
            self: The object itself.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Indicates if the attention is used in a decoder. Defaults to False.
            bias (bool, optional): Indicates if bias is added to the linear transformations. Defaults to True.
            is_causal (bool, optional): Indicates if the attention is causal. Defaults to False.
            config (Optional[SeamlessM4Tv2Config], optional): The configuration for the attention. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
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

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, projection: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to reshape the input projection tensor to match the specified number of heads and head dimension.

        Args:
            self (SeamlessM4Tv2Attention): The instance of the SeamlessM4Tv2Attention class.
            projection (mindspore.Tensor): The input projection tensor that needs to be reshaped.

        Returns:
            mindspore.Tensor: A new tensor with the reshaped projection based on the specified number of heads
                and head dimension.

        Raises:
            None.
        """
        new_projection_shape = projection.shape[:-1] + (self.num_heads, self.head_dim)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]

        # use encoder_hidden_states if cross attention
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # checking that the `sequence_length` of the `past_key_value` is the same as the he provided
        # `encoder_hidden_states` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k_proj(current_states))
            value_states = self._shape(self.v_proj(current_states))
            if past_key_value is not None and not is_cross_attention:
                # reuse k, v, self_attention
                key_states = ops.cat([past_key_value[0], key_states], axis=2)
                value_states = ops.cat([past_key_value[1], value_states], axis=2)

        query_states = self._shape(self.q_proj(hidden_states) * self.scaling)
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = ops.softmax(attention_scores.float(), axis=-1).type_as(attention_scores)
        attn_weights = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        #  attn_output = torch.bmm(attn_probs, value_states) ?
        context_states = ops.matmul(attn_weights, value_states)
        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim) ?
        context_states = context_states.permute(0, 2, 1, 3).view(batch_size, seq_length, -1)
        attn_output = self.out_proj(context_states)

        if output_attentions:
            return attn_output, attn_weights, past_key_value
        return attn_output, None, past_key_value


# Copied from transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeDenseActDense with NllbMoe->SeamlessM4Tv2,DenseActDense->FeedForwardNetwork, d_model->hidden_size
class SeamlessM4Tv2FeedForwardNetwork(nn.Cell):

    """
    The SeamlessM4Tv2FeedForwardNetwork class represents a feedforward neural network for the SeamlessM4Tv2 model.
    It inherits from nn.Cell and contains methods for initializing the network and constructing the forward pass.

    Attributes:
        config (SeamlessM4Tv2Config): The configuration object for the SeamlessM4Tv2 model.
        ffn_dim (int): The dimension of the feedforward network.

    Methods:
        __init__: Initializes the feedforward network with the given configuration and dimension.
        construct: Constructs the forward pass of the feedforward network using the given hidden states.

    Example:
        ```python
        >>> # Instantiate the feedforward network
        >>> config = SeamlessM4Tv2Config()
        >>> ffn_dim = 512
        >>> ffn_network = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim)
        ...
        >>> # Perform forward pass
        >>> hidden_states = ...
        >>> output = ffn_network.construct(hidden_states)
        ```
    """
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int):
        """
        Initializes the SeamlessM4Tv2FeedForwardNetwork.

        Args:
            self: The object itself.
            config (SeamlessM4Tv2Config): An instance of SeamlessM4Tv2Config containing the configuration parameters
                for the feed forward network.
            ffn_dim (int): The dimensionality of the feed forward network.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If any of the input parameters are out of valid range or not as expected.
        """
        super().__init__()
        self.fc1 = nn.Dense(config.hidden_size, ffn_dim)
        self.fc2 = nn.Dense(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(p=config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def construct(self, hidden_states):
        """
        This method constructs the feed-forward network for the SeamlessM4Tv2FeedForwardNetwork class.

        Args:
            self (object): The instance of the SeamlessM4Tv2FeedForwardNetwork class.
            hidden_states (mindspore.Tensor): The hidden states input to the network.

        Returns:
            mindspore.Tensor: The output tensor after passing through the feed-forward network.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the dimensions or types of the input parameters are not compatible with the network.
            RuntimeError: If there is an issue during the execution of the feed-forward network.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.fc2.weight, mindspore.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and self.fc2.weight.dtype not in (mindspore.int8, mindspore.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoderLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2EncoderLayer(nn.Cell):

    """
    This class represents an encoder layer for the SeamlessM4Tv2 model. It inherits from the nn.Cell class.

    The encoder layer performs multi-head self-attention and feed-forward network operations on the input hidden states.

    Attributes:
        embed_dim (int): The dimension of the hidden states.
        self_attn (SeamlessM4Tv2Attention): The self-attention module for the encoder layer.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization for the hidden states after self-attention.
        ffn (SeamlessM4Tv2FeedForwardNetwork): The feed-forward network module for the encoder layer.
        ffn_layer_norm (nn.LayerNorm): Layer normalization for the hidden states after feed-forward network.
        ffn_dropout (nn.Dropout): Dropout layer for the feed-forward network output.

    Methods:
        construct(hidden_states, attention_mask, output_attentions=False):
            Performs the forward pass of the encoder layer.

            Args:

            - hidden_states (mindspore.Tensor): Input hidden states of shape `(batch, seq_len, embed_dim)`.
            - attention_mask (mindspore.Tensor): Attention mask of size `(batch, 1, tgt_len, src_len)` where padding
            elements are indicated by very large negative values.

                - output_attentions (bool, optional): Whether to output attention weights. Defaults to False.

            Returns:

            - outputs (tuple): A tuple containing the computed hidden states.
            If output_attentions=True, the tuple also contains attention weights.
    """
    def __init__(self, config: SeamlessM4Tv2Config, encoder_ffn_dim=None, encoder_attention_heads=None):
        """
        Initializes a new instance of the SeamlessM4Tv2EncoderLayer class.

        Args:
            self: The object itself.
            config (SeamlessM4Tv2Config): An instance of the SeamlessM4Tv2Config class containing the
                configuration settings.
            encoder_ffn_dim (int, optional): The dimension of the feed-forward network in the encoder.
                If not provided, it will default to the value specified in the config.
            encoder_attention_heads (int, optional): The number of attention heads in the encoder.
                If not provided, it will default to the value specified in the config.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm([config.hidden_size])
        self.ffn_dropout = nn.Dropout(p=config.activation_dropout)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        output_attentions: bool = False,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoderLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2DecoderLayer(nn.Cell):

    """
    This class represents a decoder layer of the SeamlessM4Tv2 model. It is used to process the input hidden states
    and generate the output hidden states for the decoder part of the model.

    Attributes:
        `embed_dim`: The dimension of the hidden states.
        `self_attn`: The self-attention mechanism used in the decoder layer.
        `dropout`: The dropout probability used in the decoder layer.
        `activation_fn`: The activation function used in the decoder layer.
        `attn_dropout`: The dropout probability used in the self-attention mechanism.
        `self_attn_layer_norm`: The layer normalization applied to the self-attention output.
        `cross_attention`: The cross-attention mechanism used in the decoder layer.
        `cross_attention_layer_norm`: The layer normalization applied to the cross-attention output.
        `ffn`: The feed-forward network used in the decoder layer.
        `ffn_layer_norm`: The layer normalization applied to the feed-forward network output.
        `ffn_dropout`: The dropout probability used in the feed-forward network.

    Methods:
        `construct`: Performs the forward pass of the decoder layer.

    Args:
        `hidden_states (mindspore.Tensor)`: The input hidden states of shape `(batch, seq_len, embed_dim)`.
        `attention_mask (mindspore.Tensor)`: The attention mask of size `(batch, 1, tgt_len, src_len)`
            where padding elements are indicated by very large negative values.
        `encoder_hidden_states (mindspore.Tensor)`:
            The cross-attention input hidden states of shape `(batch, seq_len, embed_dim)`.
        `encoder_attention_mask (mindspore.Tensor)`: The encoder attention mask of size `(batch, 1, tgt_len, src_len)`
            where padding elements are indicated by very large negative values.
        `past_key_value (Tuple(mindspore.Tensor))`: The cached past key and value projection states.
        `output_attentions (bool, optional)`: Whether or not to return the attentions tensors of all attention layers.
        `use_cache (bool, optional)`: Whether or not to use the cached key and value projection states.

    Returns:
        `outputs`: A tuple containing the output hidden states and the present key and value projection states.
            If `output_attentions` is `True`, the tuple also contains the self-attention weights and the
            cross-attention weights.

    Note:
        The attention weights are returned only if `output_attentions` is `True`.
    """
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        """
        Initialize a decoder layer in the SeamlessM4Tv2 model.

        Args:
            self: The object instance.
            config (SeamlessM4Tv2Config): The configuration object for the SeamlessM4Tv2 model.
            decoder_ffn_dim (int, optional): The dimension of the feed-forward network in the decoder layer.
                Defaults to None.
            decoder_attention_heads (int, optional): The number of attention heads to use in the decoder layer.
                Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(p=config.dropout)

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.cross_attention = SeamlessM4Tv2Attention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.cross_attention_layer_norm = nn.LayerNorm([self.embed_dim])

        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm([config.hidden_size])
        self.ffn_dropout = nn.Dropout(p=config.activation_dropout)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`):
                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by
                very large negative values.
            past_key_value (`Tuple(mindspore.Tensor)`):
                cached past key and value projection states
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
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=cross_attn_past_key_value,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value += cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class SeamlessM4Tv2TextToUnitDecoderLayer(nn.Cell):

    """
    This class represents a layer of the SeamlessM4Tv2TextToUnitDecoder, which is used for converting text input into
    unit representations. It inherits from the nn.Cell class.

    Attributes:
        dropout (float): The dropout probability.
        embed_dim (int): The dimension of the input embedding.
        self_attn (SeamlessM4Tv2Attention): The self-attention mechanism.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization for self-attention output.
        conv1 (nn.Conv1d): The first convolutional layer.
        activation_fn (function): The activation function.
        conv2 (nn.Conv1d): The second convolutional layer.
        conv_layer_norm (nn.LayerNorm): The layer normalization for convolutional output.
        conv_dropout (nn.Dropout): The dropout layer for the convolutional output.

    Methods:
        construct(hidden_states, attention_mask=None, padding_mask=None, output_attentions=False):
            Constructs the layer.

            Args:

            - hidden_states (mindspore.Tensor): The input to the layer of shape (batch, seq_len, embed_dim).
            - attention_mask (mindspore.Tensor, optional): The attention mask of size (batch, 1, tgt_len, src_len)
            where padding elements are indicated by very large negative values.
            - padding_mask (mindspore.Tensor, optional): Indicates which inputs are to be ignored due to padding,
            where elements are either 1 for not masked or 0 for masked.
            - output_attentions (bool, optional): Whether or not to return the attention tensors of all attention
            layers. Default is False.

            Returns:

            - outputs: A tuple containing the hidden states and present key-value tensors. If output_attentions is True,
            it also includes the attention weights tensors.

    Note:
        - The hidden_states tensor is passed through the self-attention mechanism, followed by a residual connection
        and layer normalization.
        - If padding_mask is provided, the hidden_states tensor is masked before applying the first convolutional layer.
        - The hidden_states tensor is then passed through the first convolutional layer, followed by an activation
        function, a second convolutional layer, and dropout.
        - The output of the second convolutional layer is added to the residual tensor from the self-attention mechanism,
        followed by layer normalization.
        - The final output is returned as a tuple, including the hidden states and present key-value tensors.
        If output_attentions is True, the attention weights tensors are also included.
    """
    def __init__(self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None):
        """
        Initializes an instance of the `SeamlessM4Tv2TextToUnitDecoderLayer` class.

        Args:
            self: The object itself.
            config (SeamlessM4Tv2Config): An instance of the `SeamlessM4Tv2Config` class containing
                configuration settings.
            decoder_ffn_dim (int, optional): The dimension of the feed-forward network in the decoder.
                If not provided, it takes the value from `config.decoder_ffn_dim`.
            decoder_attention_heads (int, optional): The number of attention heads in the decoder.
                If not provided, it takes the value from `config.decoder_attention_heads`.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )
        self.dropout = config.dropout
        self.embed_dim = config.hidden_size

        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, pad_mode="same")
        self.activation_fn = ACT2FN[config.activation_function]
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=7, stride=1, pad_mode="same")

        self.conv_layer_norm = nn.LayerNorm([config.hidden_size])
        self.conv_dropout = nn.Dropout(p=self.dropout)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        padding_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            padding_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Conv
        residual = hidden_states

        # Apply padding mask to avoid leaking padded positions in the convolution layer
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.conv2(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        hidden_states = self.conv_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.conv_layer_norm(hidden_states)

        outputs = (hidden_states, present_key_value)

        if output_attentions:
            outputs += self_attn_weights

        return outputs


############ SUB-MODELS related code ################


class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SeamlessM4Tv2Config
    base_model_prefix = "seamless_m4t_v2"
    _no_split_modules = [
        "SeamlessM4Tv2EncoderLayer",
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
        "SeamlessM4Tv2TextToUnitDecoderLayer",
    ]

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, SeamlessM4Tv2ConformerSelfAttention):
            if hasattr(cell, "pos_bias_u"):
                cell.pos_bias_u.set_data(initializer(XavierUniform(),
                                                    cell.pos_bias_u.shape, cell.pos_bias_u.dtype))
            if hasattr(cell, "pos_bias_v"):
                cell.pos_bias_v.set_data(initializer(XavierUniform(),
                                                    cell.pos_bias_v.shape, cell.pos_bias_v.dtype))

        elif isinstance(cell, SeamlessM4Tv2ConformerFeatureProjection):
            k = math.sqrt(1 / cell.projection.in_channels)
            cell.projection.weight.set_data(initializer(Uniform(k),
                                        cell.projection.weight.shape, cell.projection.weight.dtype))
            cell.projection.bias.set_data(initializer(Uniform(k),
                                        cell.projection.bias.shape, cell.projection.bias.dtype))

        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            cell.weight.set_data(initializer(HeNormal(),
                                              cell.weight.shape, cell.weight.dtype))

            if cell.bias is not None:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(Uniform(k),
                                   cell.bias.shape, cell.bias.dtype))

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TPreTrainedModel._compute_sub_sample_lengths_from_attention_mask
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        '''
        Compute the sub-sample lengths from the attention mask.

        Args:
            self (SeamlessM4Tv2PreTrainedModel): The instance of the SeamlessM4Tv2PreTrainedModel class.
            attention_mask (numpy.ndarray): The attention mask for the input sequence. It is a 2D array representing
                the mask with shape (batch_size, sequence_length).

        Returns:
            numpy.ndarray: An array of sub-sample lengths computed based on the attention mask.
                It has the same shape as attention_mask, containing the sub-sample lengths for each sequence
                in the batch.

        Raises:
            TypeError: If the input attention_mask is not a numpy array.
            ValueError: If the input attention_mask has an invalid shape or contains invalid values.
        '''
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = kernel_size // 2
        seq_lens = attention_mask.shape[1] - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.astype(mindspore.float32).floor()

    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        """
        if not hasattr(self.generation_config, "id_to_text"):
            raise ValueError(
                """This model generation config doesn't have a `id_to_text` key which maps
                token ids to subwords. Make sure to load the right generation config."""
            )
        batch_size, sequence_len = input_ids.shape

        subwords_batch = []
        for batch_id in range(batch_size):
            subwords = []
            for i in range(sequence_len):
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    def _count_character_length_in_subword(
        self,
        input_ids,
        subwords_batch,
        merge_space_with_prev_subword=False,
        pad_token_id=0,
        unk_token_id=1,
        space="▁",
    ):
        """
        Counts the number of characters per text string associated with the input token id.

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            merge_space_with_prev_subword (`bool`, *optional*, defaults to `False`):
                Indicates if the space character is merged with the previous subword. If `False`, it will be merged
                with the next subword.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
            space (`str`, *optional*, defaults to `"▁"`):
                The space character.
        """
        batch_size, _ = input_ids.shape

        char_count_per_id = input_ids.new_zeros(input_ids.shape)

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            # We slice out the tensor till the padding index.
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]

            is_next_start_with_space = [
                len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < len(subwords) - 1 else False
                for i in range(len(subwords))
            ]
            is_punc = [
                len(subwords[i]) == 1
                and not subwords[i].isalpha()
                and not subwords[i].isnumeric()
                and subwords[i] != space
                for i in range(len(subwords))
            ]
            for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
                if subword_idx == pad_token_id:
                    break

                if subword_idx == unk_token_id:
                    # We set char_len to 1 for an unk token.
                    char_len = 1

                    if merge_space_with_prev_subword and is_next_start_with_space[i]:
                        char_len += 1
                else:
                    # By default, spaces are merged with the next subword.
                    # char_len includes the space.
                    char_len = len(subword)

                    if merge_space_with_prev_subword:
                        # Add the space for the next subword.
                        if is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the current subword.
                        if i > 0 and is_next_start_with_space[i - 1]:
                            char_len -= 1
                    else:
                        # Merge space with punctuation mark by default.
                        if is_punc[i] and is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the subword succeeding the punctuation mark.
                        elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                            char_len -= 1

                char_count_per_id[batch_id, i] = char_len

        return char_count_per_id

    def _get_char_input_ids(self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1):
        """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`List[List[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `mindspore.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
        if not hasattr(self.generation_config, "char_to_id"):
            raise ValueError(
                """This model generation config doesn't have a `char_to_id` key which maps
                characters to character ids. Make sure to load the right generation config."""
            )

        batch_size = input_ids.shape[0]
        max_len = int(char_count_per_id.sum(1).max().item())

        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill(pad_token_id)

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            total = 0
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    char_ids = [unk_token_id]
                else:
                    # Get char token indices corresponding to the subwords.
                    char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
                char_seq_len = len(char_ids)
                char_seqs[batch_id, total : total + char_seq_len] = mindspore.tensor(char_ids).to(char_seqs.dtype)
                total += char_seq_len
        return char_seqs

    def _hard_upsample(self, hidden_states, durations):
        """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
        if hidden_states.shape[0] == 1:
            hidden_states = ops.repeat_interleave(hidden_states, durations.view(-1), axis=1)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning_once(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                ops.repeat_interleave(hidden_state, duration, axis=0)
                for (hidden_state, duration) in zip(hidden_states, durations)
            ]

            hidden_states = pad_sequence(hidden_states, batch_first=True)

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TSpeechEncoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):

    """
    This class represents a speech encoder model for the SeamlessM4Tv2 architecture.
    It is a subclass of SeamlessM4Tv2PreTrainedModel.

    The SeamlessM4Tv2SpeechEncoder class initializes various components required for the speech encoding process,
    such as feature projection, encoder, feed-forward network, adapter, and layer normalization.

    The class provides a construct method that takes input features and optional parameters like attention mask,
    output attentions, output hidden states, and return dictionary flag. It processes the input
    features through the feature projection, encoder, feed-forward network, adapter (if available),
    and layer normalization to produce the encoded speech representation. The method returns the encoded speech
    representation along with other encoder outputs, such as hidden states and attentions, as a named tuple called
    Wav2Vec2BaseModelOutput.

    Note:
        The class assumes that either the input features or the inputs embeddings are not None.
        If both are None, a ValueError is raised.

    For more details on the SeamlessM4Tv2 architecture and its components,
    please refer to the SeamlessM4Tv2 documentation.
    """
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4Tv2Config):
        """
        Initializes a SeamlessM4Tv2SpeechEncoder object.

        Args:
            self: The instance of the SeamlessM4Tv2SpeechEncoder class.
            config (SeamlessM4Tv2Config): An object of type SeamlessM4Tv2Config containing configuration parameters.
                The config object is used to initialize various components within the encoder.
                It must be an instance of SeamlessM4Tv2Config class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=0.0)
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm([config.hidden_size])

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_features: Optional[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        """
        Constructs the SeamlessM4Tv2SpeechEncoder.

        Args:
            self (SeamlessM4Tv2SpeechEncoder): An instance of the SeamlessM4Tv2SpeechEncoder class.
            input_features (Optional[mindspore.Tensor]): The input features for the encoder. It can be None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder. It can be None.
            output_attentions (Optional[bool]): Whether to include attentions in the output.
                If not provided, it uses the default value from the configuration.
            output_hidden_states (Optional[bool]): Whether to include hidden states in the output.
                If not provided, it uses the default value from the configuration.
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple.
                If not provided, it uses the default value from the configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Tuple, Wav2Vec2BaseModelOutput]:
                The output of the SeamlessM4Tv2SpeechEncoder.

                If return_dict is False, it returns a tuple containing the hidden states and other encoder outputs. If
                return_dict is True, it returns a Wav2Vec2BaseModelOutput object containing the hidden states,
                hidden states from the encoder, and attentions from the encoder.

        Raises:
            ValueError: If both input_features and inputs_embeds are None in SeamlessM4Tv2SpeechEncoder.forward.
                Make sure one of them is not None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        hidden_states = self.feature_projection(input_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        hidden_states = self.inner_layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# inspired from MBart and NllbMoe
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2Encoder(SeamlessM4Tv2PreTrainedModel):

    """
    World Class Technical Documentation for SeamlessM4Tv2Encoder:

    The `SeamlessM4Tv2Encoder` class is a Python class that represents an encoder module in the SeamlessM4Tv2 model.
    This class inherits from the `SeamlessM4Tv2PreTrainedModel` class.

    Summary:
        The `SeamlessM4Tv2Encoder` class implements the encoder module of the SeamlessM4Tv2 model.
        It takes input tokens, applies embedding and positional encoding, and passes it through multiple encoder layers
        to generate encoded representations of the input.

    Constructor:
        ```python
        >>> def __init__(self, config: SeamlessM4Tv2Config, embed_tokens: Optional[nn.Embedding] = None, is_t2u_encoder: bool = False):
        >>>     super().__init__(config)
        >>>     # Initializes parameters and attributes of the encoder
        ...
        >>>     self.post_init()
        ```

    Methods:
        construct
    """
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        """
        Initializes a new instance of the SeamlessM4Tv2Encoder class.

        Args:
            self (SeamlessM4Tv2Encoder): The instance of the class.
            config (SeamlessM4Tv2Config): The configuration object containing various settings.
            embed_tokens (Optional[nn.Embedding]): An optional pre-trained embedding layer.
            is_t2u_encoder (bool): A boolean value indicating whether the encoder is used for T2U (text-to-unit) conversion.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        if not self.is_t2u_encoder:
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4Tv2EncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.CellList(layers)

        self.layer_norm = nn.LayerNorm([config.hidden_size])

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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

        if input_ids is not None and self.is_t2u_encoder:
            raise ValueError(
                "You cannot pass input_ids to the encoder of the text_to_units model. Pass inputs_embeds instead."
            )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if not self.is_t2u_encoder:
            embed_pos = self.embed_positions(input)

            hidden_states = inputs_embeds + embed_pos
        else:
            hidden_states = inputs_embeds

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
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


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2Decoder(SeamlessM4Tv2PreTrainedModel):

    """
    A Python class representing the SeamlessM4Tv2Decoder module of the SeamlessM4Tv2 model architecture.

    This class inherits from the SeamlessM4Tv2PreTrainedModel class and implements the decoder component of the
    SeamlessM4Tv2 model. It consists of multiple decoder layers and includes functionality for embedding tokens,
    calculating positional embeddings, and performing self-attention and cross-attention operations.

    Attributes:
        config (SeamlessM4Tv2Config): The configuration object for the SeamlessM4Tv2Decoder module.
        dropout (float): The dropout probability for the decoder layers.
        layerdrop (float): The layer dropout probability for the decoder layers.
        padding_idx (int): The index of the padding token in the vocabulary.
        vocab_size (int): The size of the vocabulary.
        max_target_positions (int): The maximum number of target positions.
        embed_scale (float): The scale factor for the embedding layer.
        embed_tokens (nn.Embedding): The embedding layer for the input tokens.
        embed_positions (SeamlessM4Tv2SinusoidalPositionalEmbedding): The positional embedding layer.
        layers (nn.CellList): The list of decoder layers.
        layer_norm (nn.LayerNorm): The layer normalization module.

    Methods:
        __init__: Initializes the SeamlessM4Tv2Decoder module.
        get_input_embeddings: Returns the input embeddings.
        set_input_embeddings: Sets the input embeddings.
        construct: Constructs the SeamlessM4Tv2Decoder module.

    Please refer to the documentation of the parent class, SeamlessM4Tv2PreTrainedModel, for more details on the
    inherited attributes and methods.
    """
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        """Initialize the SeamlessM4Tv2Decoder.

        Args:
            self: The object itself.
            config (SeamlessM4Tv2Config): An instance of SeamlessM4Tv2Config containing configuration parameters
                for the decoder.
            embed_tokens (Optional[nn.Embedding]): An optional instance of nn.Embedding for token embedding.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not an instance of SeamlessM4Tv2Config.
            ValueError: If the embed_tokens parameter is not None and is not an instance of nn.Embedding.
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # if embed_tokens defined, use its shape instead
            self.embed_tokens = nn.Embedding(embed_tokens.vocab_size, embed_tokens.embedding_size, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2DecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.CellList(layers)
        self.layer_norm = nn.LayerNorm([config.hidden_size])

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the SeamlessM4Tv2Decoder.

        Args:
            self (SeamlessM4Tv2Decoder): An instance of the SeamlessM4Tv2Decoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2Decoder.

        Args:
            self (SeamlessM4Tv2Decoder): The instance of the SeamlessM4Tv2Decoder class.
            value: The input embeddings to be set. This should be a tensor or an instance of the Embedding class.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
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
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`mindspore.Tensor` of
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
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[3],)

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


class SeamlessM4Tv2TextToUnitDecoder(SeamlessM4Tv2PreTrainedModel):

    '''
    A decoder module for SeamlessM4Tv2 model that converts character-level hidden states into unit-level hidden states.

    This class inherits from SeamlessM4Tv2PreTrainedModel and implements methods for initializing the decoder,
    getting input embeddings, setting input embeddings, and constructing the decoder output from
    character-level inputs.

    Attributes:
        config: SeamlessM4Tv2Config
            The configuration for the SeamlessM4Tv2 model.

    Methods:
        __init__:
            Initializes the decoder with the provided configuration and optional embedding tokens.

        get_input_embeddings:
            Returns the input embeddings for the decoder.

        set_input_embeddings:
            Sets the input embeddings for the decoder.

        construct:
            Constructs the decoder output from character-level inputs including character indices, encoder hidden states,
            and optional return configurations.

    Args:
        char_input_ids (`mindspore.Tensor` of shape `(batch_size, char_sequence_length)`):
            Character indices for input sequences.
        char_count_per_id (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length)`):
            Number of characters per text input id.
        encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
            Sequence of hidden states from the encoder.
        output_attentions (`bool`, *optional*):
            Whether to return the attention tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether to return a `utils.ModelOutput` instead of a plain tuple.

    Returns:
        Union[Tuple, SeamlessM4Tv2TextToUnitDecoderOutput]: The decoder output including hidden states, attentions,
            and padding mask.
    '''
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        """
        Initializes an instance of the 'SeamlessM4Tv2TextToUnitDecoder' class.

        Args:
            self: The current object instance.
            config (SeamlessM4Tv2Config): An instance of the 'SeamlessM4Tv2Config' class containing the
                configuration settings.
            embed_tokens (Optional[nn.Embedding]): An optional instance of the 'nn.Embedding' class representing
                embedded tokens. Default is None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # if embed_tokens defined, use its shape instead
            self.embed_tokens = nn.Embedding(embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx)
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        self.embed_char = nn.Embedding(config.char_vocab_size, config.hidden_size)
        self.embed_char_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        self.pos_emb_alpha_char = Parameter(ops.ones(1))
        self.pos_emb_alpha = Parameter(ops.ones(1))
        self.duration_predictor = SeamlessM4Tv2VariancePredictor(
            config.variance_predictor_embed_dim,
            config.variance_predictor_hidden_dim,
            config.variance_predictor_kernel_size,
            config.variance_pred_dropout,
        )

        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2TextToUnitDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.CellList(layers)
        self.layer_norm = nn.LayerNorm([config.hidden_size])

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the SeamlessM4Tv2TextToUnitDecoder.

        Args:
            self (SeamlessM4Tv2TextToUnitDecoder): An instance of the SeamlessM4Tv2TextToUnitDecoder class.

        Returns:
            None: This method does not return any value.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2TextToUnitDecoder.

        Args:
            self (SeamlessM4Tv2TextToUnitDecoder): The instance of the SeamlessM4Tv2TextToUnitDecoder class.
            value (Any): The input embeddings to set.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
        self,
        char_input_ids: mindspore.Tensor = None,
        char_count_per_id: mindspore.Tensor = None,
        encoder_hidden_states: mindspore.Tensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SeamlessM4Tv2TextToUnitDecoderOutput]:
        r"""
        Args:
            char_input_ids (`mindspore.Tensor` of shape `(batch_size, char_sequence_length)`):
                Character indices. The correspondence between characters and indices can be found in `char_to_id`, a
                dictionary in the generation configuration.
            char_count_per_id (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length)`):
                Number of characters per text input id.
            encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
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

        # create padding mask for character lengths
        char_padding_mask = _compute_new_attention_mask(char_input_ids, char_count_per_id.sum(1))

        # upsample hidden states according to characters sequence lengths
        char_hidden_states = self._hard_upsample(encoder_hidden_states, char_count_per_id)
        # embed char positions
        char_positions = self.pos_emb_alpha_char * self.embed_char_positions(inputs_embeds=char_hidden_states)
        # update char hidden states with positions and char embeddings
        char_hidden_states = self.embed_char(char_input_ids) * self.embed_scale + char_positions + char_hidden_states

        # predict duration
        log_dur_pred = self.duration_predictor(char_hidden_states, padding_mask=char_padding_mask)
        dur_out = ops.clamp(ops.round((ops.exp(log_dur_pred) - 1)).long(), min=1)
        dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0.0)

        # upsample char hidden states according to predicted duration
        char_hidden_states = self._hard_upsample(char_hidden_states, dur_out)

        positions = self.pos_emb_alpha * self.embed_positions(inputs_embeds=char_hidden_states)
        hidden_states = char_hidden_states + positions

        padding_mask = _compute_new_attention_mask(hidden_states, dur_out.sum(1))
        attention_mask = _prepare_4d_attention_mask(padding_mask, hidden_states.dtype)

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for _, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns, padding_mask] if v is not None)
        return SeamlessM4Tv2TextToUnitDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            padding_mask=padding_mask,
        )


class SeamlessM4Tv2TextToUnitModel(SeamlessM4Tv2PreTrainedModel):

    """
    This class represents the SeamlessM4Tv2TextToUnitModel, which is a Python class that inherits from the
    SeamlessM4Tv2PreTrainedModel class. The SeamlessM4Tv2TextToUnitModel is a model that performs text-to-unit
    conversion using the SeamlessM4Tv2 architecture.

    The class has two main attributes:

    - encoder: An instance of the SeamlessM4Tv2Encoder class, which is responsible for encoding the input text.
    - decoder: An instance of the SeamlessM4Tv2TextToUnitDecoder class, which is responsible for decoding the encoded
    text into unit representations.

    The SeamlessM4Tv2TextToUnitModel class provides a constructor '__init__' that takes two parameters:

    - config: An object of type SeamlessM4Tv2Config, which contains the configuration settings for the model.
    - embed_tokens_decoder (optional): An optional instance of the nn.Embedding class, which represents the
    embedding tokens for the decoder. If not provided, the default value is None.

    The class also provides a method 'construct' that is used to perform the text-to-unit conversion.
    This method takes several parameters:

    - input_ids (optional): An optional mindspore.Tensor object representing the input text IDs.
    - char_input_ids: A mindspore.Tensor object representing the character input IDs.
    - char_count_per_id: A mindspore.Tensor object representing the count of characters per input ID.
    - attention_mask (optional): An optional mindspore.Tensor object representing the attention mask.
    - encoder_outputs (optional): An optional tuple of mindspore.Tensor objects representing the encoder outputs.
    - inputs_embeds (optional): An optional mindspore.Tensor object representing the embedded inputs.
    - output_attentions (optional): An optional boolean indicating whether to output attentions.
    If not provided, the default value is None.
    - output_hidden_states (optional): An optional boolean indicating whether to output hidden states.
    If not provided, the default value is None.
    - return_dict (optional): An optional boolean indicating whether to return a dictionary.
    If not provided, the default value is None.

    The 'construct' method returns either a tuple of mindspore.Tensor objects or an instance of the Seq2SeqModelOutput
    class, depending on the value of the 'return_dict' parameter.

    Note:
        This docstring does not include signatures or any other code.
    """
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitModel.__init__ with SeamlessM4T->SeamlessM4Tv2, Decoder->TextToUnitDecoder
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        """
        Initializes a new instance of the SeamlessM4Tv2TextToUnitModel class.

        Args:
            self: The instance of the class.
            config (SeamlessM4Tv2Config): An object of type SeamlessM4Tv2Config representing the configuration
                settings for the model.
            embed_tokens_decoder (Optional[nn.Embedding]): An optional neural network embedding layer used for
                decoding tokens. Defaults to None if not provided.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.encoder = SeamlessM4Tv2Encoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4Tv2TextToUnitDecoder(config, embed_tokens_decoder)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        char_input_ids: mindspore.Tensor = None,
        char_count_per_id: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        """
        Construct the model for converting text to unit in the SeamlessM4Tv2TextToUnitModel class.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]): Input tensor representing tokenized input text IDs. Default is None.
            char_input_ids (mindspore.Tensor): Input tensor representing character-level token IDs. Default is None.
            char_count_per_id (mindspore.Tensor): Tensor containing the count of characters per token ID.
            attention_mask (Optional[mindspore.Tensor]): Tensor representing attention mask for input IDs. Default is None.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]]): Tuple containing encoder outputs. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): Tensor representing embedded inputs. Default is None.
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Flag to indicate whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag to indicate whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
                The model output containing the hidden states and attentions.

        Raises:
            None
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn, padding_mask)
        decoder_outputs = self.decoder(
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            encoder_hidden_states=encoder_outputs[0],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            padding_mask=decoder_outputs.padding_mask,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SeamlessM4Tv2TextToUnitForConditionalGeneration(SeamlessM4Tv2PreTrainedModel):

    """
    This class represents a SeamlessM4Tv2TextToUnitForConditionalGeneration model for generating conditional 
    text-to-unit outputs. It is a subclass of SeamlessM4Tv2PreTrainedModel.

    Attributes:
        model (SeamlessM4Tv2TextToUnitModel): The underlying text-to-unit model.
        lm_head (nn.Dense): The linear layer for generating the language model logits.

    Methods:
        __init__: Initializes the SeamlessM4Tv2TextToUnitForConditionalGeneration instance.
        get_encoder: Returns the encoder of the underlying model.
        get_decoder: Returns the decoder of the underlying model.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        get_input_embeddings: Returns the input embeddings of the decoder.
        set_input_embeddings: Sets the input embeddings of the decoder.
        construct: Constructs the model and returns the generated outputs.
        _tie_weights(): Ties the input and output embeddings if specified in the configuration.
    """
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        """
        Initialize the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (SeamlessM4Tv2Config): An instance of SeamlessM4Tv2Config containing configuration parameters.
            embed_tokens_decoder (Optional[nn.Embedding]): An optional nn.Embedding layer for token decoding.

        Returns:
            None.

        Raises:
            None.
        """
        # update config - used principaly for bos_token_id etc.
        config = copy.deepcopy(config)
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        super().__init__(config)

        self.model = SeamlessM4Tv2TextToUnitModel(config, embed_tokens_decoder)

        self.lm_head = nn.Dense(config.hidden_size, config.t2u_vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.get_encoder
    def get_encoder(self):
        """
        This method returns the encoder of the SeamlessM4Tv2TextToUnitForConditionalGeneration model.

        Args:
            self: The instance of the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Returns:
            encoder: This method returns the encoder of the model.

        Raises:
            None.
        """
        return self.model.encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.get_decoder
    def get_decoder(self):
        """
        Returns the decoder of the SeamlessM4Tv2TextToUnitForConditionalGeneration model.

        Args:
            self: The instance of the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Returns:
            None

        Raises:
            None
        """
        return self.model.decoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        """
        Retrieve the output embeddings for the SeamlessM4Tv2TextToUnitForConditionalGeneration model.

        Args:
            self: An instance of the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Returns:
            lm_head: The method returns the output embeddings represented by the 'lm_head'.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Args:
            self (SeamlessM4Tv2TextToUnitForConditionalGeneration):
                An instance of the SeamlessM4Tv2TextToUnitForConditionalGeneration class.
            new_embeddings: The new embeddings to set for the output.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        """
        Returns the input embeddings for the SeamlessM4Tv2TextToUnitForConditionalGeneration model.

        Args:
            self (SeamlessM4Tv2TextToUnitForConditionalGeneration): An instance of the
                SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Args:
            self (SeamlessM4Tv2TextToUnitForConditionalGeneration): The instance of the class.
            value (Any): The new input embeddings to be set for the decoder.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        char_input_ids: mindspore.Tensor = None,
        char_count_per_id: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """
        This method constructs the SeamlessM4Tv2TextToUnitForConditionalGeneration model for conditional generation.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor): The input tensor of token indices for the model.
            char_input_ids (mindspore.Tensor): The input tensor of character indices for the model.
            char_count_per_id (mindspore.Tensor): The tensor representing the count of characters per token.
            attention_mask (Optional[mindspore.Tensor]): The tensor representing the attention mask for the input.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]]): The encoder outputs to be used in the model.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs to the model.
            labels (Optional[mindspore.Tensor]): The tensor representing the labels for training.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            return_dict (Optional[bool]): Flag indicating whether to return a dict of outputs.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
                The output of the model, which can be either a Seq2SeqLMOutput object or a tuple of tensors.

        Raises:
            NotImplementedError: If the method is not fully implemented.
            ValueError: If an invalid configuration is provided.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            char_input_ids=char_input_ids,
            char_count_per_id=char_count_per_id,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return SeamlessM4Tv2TextToUnitOutput(
            last_hidden_state=lm_logits,
            padding_mask=outputs.padding_mask,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loss=masked_lm_loss,
        )

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TTextToUnitForConditionalGeneration._tie_weights
    def _tie_weights(self) -> None:
        """
        Ties the word embeddings if the configuration specifies and the output embeddings are not None.

        Args:
            self (SeamlessM4Tv2TextToUnitForConditionalGeneration): The current instance of the
                SeamlessM4Tv2TextToUnitForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())


############ VOCODER related code ################
# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Cell):

    """
    This class represents a HiFiGAN residual block, which is used for generating high-fidelity audio waveforms.
    It inherits from the nn.Cell class.

    Attributes:
        channels (int): The number of input and output channels for the convolutional layers.
        kernel_size (int): The size of the convolutional kernel.
        dilation (tuple): A tuple of dilation factors for the convolutional layers.
        leaky_relu_slope (float): The slope for the leaky ReLU activation function.

    Methods:
        __init__:
            Initializes a HiFiGAN residual block object.

        get_padding:
            Calculates the padding size for the convolutional layers based on the kernel size and dilation factor.

        apply_weight_norm:
            Applies weight normalization to the convolutional layers in the residual block.

        remove_weight_norm:
            Removes weight normalization from the convolutional layers in the residual block.

        construct:
            Constructs the residual block by sequentially applying leaky ReLU activation, convolutional layers,
            and addition with the residual. Returns the final hidden states after passing through the residual block.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        """
        Initializes a HifiGanResidualBlock object.

        Args:
            self (HifiGanResidualBlock): An instance of the HifiGanResidualBlock class.
            channels (int): The number of input and output channels for the convolutional layers.
            kernel_size (int, optional): The size of the kernel for the convolutional layers. Defaults to 3.
            dilation (tuple, optional): A tuple of dilation factors for the convolutional layers. Defaults to (1, 3, 5).
            leaky_relu_slope (float, optional): The slope of the negative part of the leaky ReLU activation function.
                Defaults to 0.1.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.CellList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    pad_mode='pad',
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.CellList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    pad_mode='pad',
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        """
        Returns the amount of padding required for the convolution operation in the HiFi-GAN residual block.

        Args:
            self: Instance of the HifiGanResidualBlock class.
            kernel_size (int): The size of the kernel used in the convolution operation.
            dilation (int, optional): The dilation rate of the convolution operation. Defaults to 1.

        Returns:
            int: The amount of padding required for the convolution operation.

        Raises:
            TypeError:
                If kernel_size or dilation is not an integer, or if the value of dilation is less than or equal to zero.
        """
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        """
        Applies weight normalization to the convolutional layers in the HifiGanResidualBlock.

        Args:
            self: An instance of the HifiGanResidualBlock class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method applies weight normalization to the convolutional layers in the HifiGanResidualBlock.
            Weight normalization is a technique that normalizes the weights of a neural network layer to stabilize
            training and improve convergence. The method iterates over the convs1 and convs2 lists, which contain
            the convolutional layers, and applies weight normalization using the nn.utils.weight_norm function.

        Note:
            - The convs1 and convs2 lists must be populated with valid convolutional layers before calling this method.
            - Weight normalization modifies the weights of the layers in-place.

        Example:
            ```python
            >>> block = HifiGanResidualBlock()
            >>> block.apply_weight_norm()
            ```
        """
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layers in a HifiGanResidualBlock.

        Args:
            self (HifiGanResidualBlock): The instance of the HifiGanResidualBlock class.
                It represents the block containing convolutional layers with weight normalization to remove.

        Returns:
            None: This method does not return any value. It modifies the convolutional layers in place by removing
                weight normalization.

        Raises:
            None.
        """
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def construct(self, hidden_states):
        """
        Constructs a single residual block in the HifiGanResidualBlock class.

        Args:
            self (HifiGanResidualBlock): The instance of the HifiGanResidualBlock class.
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output hidden states of shape (batch_size, channels, height, width).

        Raises:
            None.
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class SeamlessM4Tv2VariancePredictor(nn.Cell):

    """
    This class represents a variance predictor for SeamlessM4Tv2 models.
    It is a subclass of nn.Cell and is used to predict variances in the SeamlessM4Tv2 model.

    Attributes:
        conv1 (nn.Conv1d): A 1-dimensional convolutional layer that maps the input embedding dimensions to
            hidden dimensions.
        activation_function (nn.ReLU): The activation function used after the first convolutional layer.
        ln1 (nn.LayerNorm): Layer normalization applied after the activation function.
        dropout_module (nn.Dropout): Dropout module used to apply dropout regularization.
        conv2 (nn.Conv1d): A second 1-dimensional convolutional layer that maps the hidden dimensions to hidden dimensions.
        ln2 (nn.LayerNorm): Layer normalization applied after the second convolutional layer.
        proj (nn.Dense): A fully connected layer that maps the hidden dimensions to a single output dimension.

    Methods:
        construct(hidden_states, padding_mask=None):
            Constructs the variance predictor by applying the necessary operations on the input hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - padding_mask (mindspore.Tensor, optional): A tensor specifying the padding positions, used for masking.
            Defaults to None.

            Returns:

            - mindspore.Tensor: The predicted variances.

    """
    def __init__(self, embed_dim, hidden_dim, kernel_size, var_pred_dropout):
        """Initializes an instance of the SeamlessM4Tv2VariancePredictor class.

        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the input embedding.
            hidden_dim (int): The dimension of the hidden layer.
            kernel_size (int): The size of the convolutional kernel.
            var_pred_dropout (float): The dropout rate for variance prediction.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.conv1 = nn.Conv1d(
            embed_dim,
            hidden_dim,
            kernel_size=kernel_size,
            pad_mode="same",
        )
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm([hidden_dim])
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            pad_mode="same",
        )
        self.ln2 = nn.LayerNorm([hidden_dim])
        self.proj = nn.Dense(hidden_dim, 1)

    def construct(self, hidden_states: mindspore.Tensor, padding_mask: mindspore.Tensor = None) -> mindspore.Tensor:
        '''
        Constructs a new tensor by applying several operations to the input tensor.

        Args:
            self (SeamlessM4Tv2VariancePredictor): An instance of the SeamlessM4Tv2VariancePredictor class.
            hidden_states (mindspore.Tensor): A tensor representing the hidden states.
            padding_mask (mindspore.Tensor, optional): A tensor representing the padding mask.
                Defaults to None.

        Returns:
            mindspore.Tensor: A tensor representing the output of the constructed operations.

        Raises:
            None
        '''
        # Input: B x T x C; Output: B x T
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.swapaxes(1, 2))
        hidden_states = self.activation_fuction(hidden_states).swapaxes(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv2(hidden_states.swapaxes(1, 2))
        hidden_states = self.activation_fuction(hidden_states).swapaxes(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(axis=2)


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4THifiGan with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2HifiGan(nn.Cell):

    """
    The `SeamlessM4Tv2HifiGan` class is a neural network model designed to convert a log-mel spectrogram into
    a speech waveform. It is specifically tailored for the SeamlessM4Tv2 configuration.

    This class inherits from `nn.Cell` and contains several layers and operations to perform the conversion.
    The main components of the class include a convolutional layer (`conv_pre`), a list of upsampling
    layers (`upsampler`), a list of residual blocks (`resblocks`), and a final convolutional layer (`conv_post`).

    The `construct` method is the main entry point of the class, which takes as input a log-mel spectrogram tensor
    and returns the corresponding speech waveform tensor. The input can be batched or un-batched, depending on the
    shape of the tensor. The shape of the input tensor should be `(batch_size, sequence_length, model_in_dim)` for
    batched spectrograms or `(sequence_length, model_in_dim)` for un-batched spectrograms. The `model_in_dim` is the
    sum of `config.unit_embed_dim`, `config.lang_embed_dim`, and `config.spkr_embed_dim`.

    The method first applies the `conv_pre` layer to the input tensor to obtain the initial hidden states. It then
    iterates over the upsampling layers (`upsampler`) and applies them to the hidden states. For each upsampling layer,
    it also applies a set of residual blocks (`resblocks`) to refine the hidden states. The number of upsampling layers
    and residual blocks depends on the configuration parameters (`config`) provided during initialization.

    After the upsampling and residual block operations, the method applies a leaky ReLU activation function to the
    hidden states. It then passes the hidden states through the final `conv_post` layer, followed by a hyperbolic
    tangent activation function (`tanh`). Finally, the method squeezes the tensor along the second dimension and
    returns the resulting waveform tensor.

    Note that the shape of the output waveform tensor will be `(batch_size, num_frames)` if the input spectrogram is
    batched, or `(num_frames,)` if the input spectrogram is un-batched.

    This class provides a powerful tool for converting log-mel spectrograms into speech waveforms, enabling applications
    such as text-to-speech synthesis and audio generation.
    """
    def __init__(self, config: SeamlessM4Tv2Config):
        """
        __init__

        Args:
            self: Instance of the SeamlessM4Tv2HifiGan class.
            config (SeamlessM4Tv2Config): An instance of SeamlessM4Tv2Config containing configuration parameters
                for the model. It includes unit_embed_dim, lang_embed_dim, spkr_embed_dim, leaky_relu_slope,
                resblock_kernel_sizes, upsample_rates, upsample_kernel_sizes, upsample_initial_channel,
                resblock_dilation_sizes.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        model_in_dim = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            pad_mode='pad',
            padding=3,
        )

        self.upsampler = nn.CellList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.Conv1dTranspose(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    pad_mode='pad',
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.CellList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, pad_mode='pad', padding=3)

    def construct(self, input_embeds: mindspore.Tensor) -> mindspore.Tensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`mindspore.Tensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                model_in_dim)`, or un-batched and of shape `(sequence_length, model_in_dim)`. Note that `model_in_dim`
                is the sum of `config.unit_embed_dim`, `config.lang_embed_dim` and `config.spkr_embed_dim`.

        Returns:
            `mindspore.Tensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        hidden_states = self.conv_pre(input_embeds)
        for i in range(self.num_upsamples):
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = ops.leaky_relu(hidden_states, 0.01)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = ops.tanh(hidden_states)

        # remove seq-len dim since this collapses to 1
        waveform = hidden_states.squeeze(1)

        return waveform


class SeamlessM4Tv2CodeHifiGan(PreTrainedModel):

    """
    This class represents the SeamlessM4Tv2CodeHifiGan model, which is used for speech synthesis and translation.
    It inherits from the PreTrainedModel class.

    Attributes:
        pad_token_id (int): The ID of the padding token in the input sequence.
        dur_predictor (SeamlessM4Tv2VariancePredictor): The variance predictor module for duration prediction.
        unit_embedding (nn.Embedding): The embedding layer for unit tokens.
        speaker_embedding (nn.Embedding): The embedding layer for speaker IDs.
        language_embedding (nn.Embedding): The embedding layer for language IDs.
        hifi_gan (SeamlessM4Tv2HifiGan): The high-fidelity generative adversarial network for speech synthesis.

    Methods:
        _get_dur_output_lengths: Computes the output length after the duration layer.
        _get_output_hifigan_lengths: Computes the output length of the hifigan convolutional layers.
        construct: Constructs the output sequence using the input tokens, speaker ID, and language ID.
        _init_weights: Initializes the weights of the model.
        apply_weight_norm: Applies weight normalization to the model.
        remove_weight_norm: Removes weight normalization from the model.
    """
    config_class = SeamlessM4Tv2Config
    main_input_name = "input_embeds"
    _no_split_modules = []

    def __init__(self, config):
        """
        Initializes an instance of SeamlessM4Tv2CodeHifiGan.

        Args:
            self: The instance of the class.
            config: A configuration object containing various settings and parameters for the model.
                It is expected to have the following attributes:

                - t2u_pad_token_id (int): The padding token ID for the model.
                - unit_embed_dim (int): The dimension of unit embeddings.
                - variance_predictor_kernel_size (int): The kernel size for the variance predictor.
                - var_pred_dropout (float): The dropout rate for the variance predictor.
                - unit_hifi_gan_vocab_size (int): The vocabulary size for unit HiFi-GAN.
                - vocoder_num_spkrs (int): The number of speakers for the vocoder.
                - spkr_embed_dim (int): The dimension of speaker embeddings.
                - vocoder_num_langs (int): The number of languages for the vocoder.
                - lang_embed_dim (int): The dimension of language embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.pad_token_id = config.t2u_pad_token_id
        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout
        self.dur_predictor = SeamlessM4Tv2VariancePredictor(embed_dim, embed_dim, kernel_size, var_pred_dropout)

        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        self.hifi_gan = SeamlessM4Tv2HifiGan(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._get_dur_output_lengths
    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        unit_lengths = (input_ids != self.pad_token_id).sum(1)

        # take care of edge cases where no padding or too many padding
        unit_lengths = ops.clamp(unit_lengths, 0, dur_out.shape[1] - 1)

        cumulative_dur_out = ops.cumsum(dur_out, axis=1)
        unit_lengths = cumulative_dur_out.gather_elements(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()

        return unit_lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._get_output_hifigan_lengths
    def _get_output_hifigan_lengths(self, input_lengths: Union[mindspore.Tensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (
                ops.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode="floor") + 1
            )

        def _swapaxes_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # conv_pre
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        # upsampler
        for _, (upsample_rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)
        ):
            input_lengths = _swapaxes_conv_out_length(
                input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
            )

        # resblock
        for _ in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(
                        input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                    )

                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

        # conv_post
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

        return input_lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.forward with SeamlessM4T->SeamlessM4Tv2, spkr_id->speaker_id
    def construct(
        self, input_ids: mindspore.Tensor, speaker_id: mindspore.Tensor, lang_id: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            speaker_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        hidden_states = self.unit_embedding(input_ids).swapaxes(1, 2)
        spkr = self.speaker_embedding(speaker_id).swapaxes(1, 2)
        lang = self.language_embedding(lang_id).swapaxes(1, 2)

        log_dur_pred = self.dur_predictor(hidden_states.swapaxes(1, 2))
        dur_out = ops.clamp(ops.round((ops.exp(log_dur_pred) - 1)).long(), min=1)
        # B x C x T
        if hidden_states.shape[0] == 1:
            hidden_states = ops.repeat_interleave(hidden_states, dur_out.view(-1), axis=2)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                ops.repeat_interleave(hidden_state, duration, axis=-1).swapaxes(0, 1)
                for (hidden_state, duration) in zip(hidden_states, dur_out)
            ]

            hidden_states = pad_sequence(hidden_states, batch_first=True).swapaxes(1, 2)

        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = ops.cat([lang, hidden_states, spkr], axis=1)

        hidden_states = self.hifi_gan(hidden_states)

        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan._init_weights
    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, (nn.Dense, nn.Conv1d, nn.Conv1dTranspose)):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.apply_weight_norm
    def apply_weight_norm(self):
        """
        Apply weight normalization to the HifiGan model layers.

        Args:
            self: Instance of the SeamlessM4Tv2CodeHifiGan class. Represents the current instance of the class.

        Returns:
            None.

        Raises:
            None: However, if any exceptions occur during the weight normalization process,
                they will be propagated up the call stack.
        """
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TCodeHifiGan.remove_weight_norm
    def remove_weight_norm(self):
        """
        Removes weight normalization from the specified layers in the HifiGan model.

        Args:
            self: An instance of the SeamlessM4Tv2CodeHifiGan class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method removes weight normalization from the following layers in the HifiGan model:

            - self.hifi_gan.conv_pre: The convolutional layer before upsampling.
            - self.hifi_gan.upsampler: A list of upsampling layers.
            - self.hifi_gan.resblocks: A list of residual blocks.
            - self.hifi_gan.conv_post: The final convolutional layer after upsampling.

        Weight normalization is a technique used to normalize the weights of neural network layers.
        By removing weight normalization, the weights of the specified layers are no longer normalized, which can have
        an impact on the performance of the model.

        Note that this method modifies the layers in-place and does not return any value.
        """
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)


############ WHOLE MODEL related code ################
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToText with SeamlessM4T->SeamlessM4Tv2,SeamlessM4Tv2Tokenizer->SeamlessM4TTokenizer,SeamlessM4Tv2Processor->SeamlessM4TProcessor
class SeamlessM4Tv2ForTextToText(SeamlessM4Tv2PreTrainedModel):

    """
    A class that represents a SeamlessM4Tv2 model for text-to-text tasks. This model is used for generating sequences
    of token IDs.

    Inherits from `SeamlessM4Tv2PreTrainedModel`.

    Attributes:
        shared (nn.Embedding): Embedding layer for shared tokens.
        text_encoder (SeamlessM4Tv2Encoder): Text encoder module.
        text_decoder (SeamlessM4Tv2Decoder): Text decoder module.
        lm_head (nn.Dense): Linear layer for language modeling head.

    Methods:
        __init__: Initializes the model with the given configuration.
        get_encoder: Returns the text encoder module.
        get_decoder: Returns the text decoder module.
        get_output_embeddings: Returns the language modeling head.
        set_output_embeddings, new_embeddings): Sets the language modeling head with new embeddings.
        get_input_embeddings: Returns the input embeddings of the text decoder.
        set_input_embeddings: Sets the input embeddings of both the text encoder and text decoder.
        _tie_weights: Ties the weights of the shared embeddings with the embeddings of the text encoder, text decoder,
            and language modeling head.
        construct: Constructs the model for text-to-text generation.
        generate: Generates sequences of token ids.
        prepare_inputs_for_generation: Prepares input tensors for text generation.

    Note:
        This class is a world-class technical documentation writer's representation of the code and may not reflect the
        actual implementation or functionality of the class.
    """
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4Tv2Config):
        """
        Initialize the SeamlessM4Tv2ForTextToText model with the provided configuration.

        Args:
            self (SeamlessM4Tv2ForTextToText): The instance of the SeamlessM4Tv2ForTextToText class.
            config (SeamlessM4Tv2Config): An object containing the configuration parameters for the model.
                This includes vocab_size (int): The size of the vocabulary.
                hidden_size (int): The size of the hidden layers.
                pad_token_id (int): The ID of the padding token.

        Returns:
            None.

        Raises:
            NotImplementedError: If any required functionality is not implemented.
            ValueError: If the configuration parameters are invalid or missing.
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        This method returns the text encoder used by the SeamlessM4Tv2ForTextToText class.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToText class.

        Returns:
            text_encoder: This method returns the text encoder used by the SeamlessM4Tv2ForTextToText class.

        Raises:
            None
        """
        return self.text_encoder

    def get_decoder(self):
        """
        This method returns the text decoder used in the SeamlessM4Tv2ForTextToText class.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToText class.

        Returns:
            text_decoder: This method returns the text decoder associated with the SeamlessM4Tv2ForTextToText instance.

        Raises:
            This method does not raise any exceptions.
        """
        return self.text_decoder

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the SeamlessM4Tv2ForTextToText model.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToText class.

        Returns:
            lm_head: The method returns the output embeddings of the model as a tensor.

        Raises:
            None.

        This method retrieves the output embeddings of the SeamlessM4Tv2ForTextToText model. The output embeddings
        represent the learned representations of the input text in a continuous vector space. These embeddings can
        be further used for downstream tasks such as text classification, information retrieval, or generation.

        Note that the return value of this method is a tensor containing the output embeddings. This tensor can be used
        for further processing or analysis, but it does not have any specific restrictions or
        limitations.

        Example:
            ```python
            >>> model = SeamlessM4Tv2ForTextToText()
            >>> embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the SeamlessM4Tv2ForTextToText model.

        Args:
            self (SeamlessM4Tv2ForTextToText): The instance of the SeamlessM4Tv2ForTextToText class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings of the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the SeamlessM4Tv2ForTextToText model.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToText class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the text decoder of the SeamlessM4Tv2ForTextToText model.
        The input embeddings are used as the initial input for the model's text-to-text translation process.

        Note that the method takes only one parameter, 'self', which refers to an instance of the
        SeamlessM4Tv2ForTextToText class. There are no restrictions on this parameter.

        The method does not raise any exceptions.

        Example:
            ```python
            >>> seamless_model = SeamlessM4Tv2ForTextToText()
            >>> embeddings = seamless_model.get_input_embeddings()
            >>> # Perform further operations with the embeddings
            ```
        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the SeamlessM4Tv2ForTextToText model.

        Args:
            self (SeamlessM4Tv2ForTextToText): The instance of the SeamlessM4Tv2ForTextToText class.
            value (torch.Tensor): The input embeddings to be set. It should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the provided input embeddings 'value' does not match the expected shape.
            AttributeError: If the 'embed_tokens' attribute is not found in the 'text_encoder' or 'text_decoder' objects.
            TypeError: If the provided 'value' is not a torch.Tensor type.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        """
        Ties the weights between the shared embeddings and the language model decoder.

        Args:
            self (SeamlessM4Tv2ForTextToText): An instance of the SeamlessM4Tv2ForTextToText class.

        Returns:
            None

        Raises:
            None

        This method ties the weights between the shared embeddings and the language model decoder if the 'tie_word_embeddings'
        configuration option is set to True. It uses the '_tie_or_clone_weights' helper method to perform the weight tying.
        The shared embedding weights are tied to the text encoder and text decoder embedding weights, and the language model
        head weights are tied to the shared embedding weights.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """
        Constructs the SeamlessM4Tv2ForTextToText model.

        Args:
            self (SeamlessM4Tv2ForTextToText): The instance of the SeamlessM4Tv2ForTextToText class.
            input_ids (mindspore.Tensor, optional): The input tensor of shape [batch_size, seq_length]
                containing the input IDs.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor of shape [batch_size, seq_length]
                containing the attention mask values.
            decoder_input_ids (mindspore.Tensor, optional): The decoder input tensor of shape [batch_size, seq_length]
                containing the decoder input IDs.
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask tensor of shape
                [batch_size, seq_length] containing the decoder attention mask values.
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs tuple containing
                the encoder hidden states, hidden states, and attentions.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values tuple containing the
                past key values.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings tensor of shape
                [batch_size, seq_length, hidden_size] containing the input embeddings.
            decoder_inputs_embeds (mindspore.Tensor, optional): The decoder input embeddings tensor of shape
                [batch_size, seq_length, hidden_size] containing the decoder input embeddings.
            labels (mindspore.Tensor, optional): The labels tensor of shape [batch_size, seq_length]
                containing the labels.
            use_cache (bool, optional): Whether to use cache for decoding.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple of outputs.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
                The model output.

                - If `return_dict` is False, it returns a tuple containing the masked language model loss, logits,
                encoder hidden states, and decoder hidden states.
                - If `return_dict` is True, it returns a Seq2SeqLMOutput object containing the loss,
                logits, past key values, decoder hidden states, decoder attentions, cross attentions, encoder last
                hidden state, encoder hidden states, and encoder attentions.

        Raises:
            None.
        """
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        encoder_attention_mask = attention_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_ids=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`mindspore.Tensor` of varying shape depending on the modality, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
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
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`:
                A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.
                The possible [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        # prepare text_decoder_input_ids
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                # also accept __xxx__
                tgt_lang = tgt_lang.replace("__", "")
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in
                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                    )
                # tgt_lang gets priority over decoder input ids
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = mindspore.tensor([[text_tgt_lang_id]] * batch_size)
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )

        return super().generate(
            input_ids,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        Args:
            self: Reference to the current instance of the class.
            decoder_input_ids (Tensor): Tensor of input IDs for the decoder.
            past_key_values (Optional[Tuple]): Tuple of past key values for the decoder. Default is None.
            attention_mask (Tensor): Tensor indicating where to pay attention to the input. Default is None.
            use_cache (bool): Flag indicating whether to use cache for generation. Default is None.
            encoder_outputs (Tensor): Tensor containing outputs from the encoder.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (None): Always set to None.
                - 'encoder_outputs' (Tensor): Outputs from the encoder.
                - 'past_key_values' (Optional[Tuple]): Past key values for the decoder.
                - 'decoder_input_ids' (Tensor): Processed decoder input IDs.
                - 'attention_mask' (Tensor): Attention mask for the input.
                - 'use_cache' (bool): Flag indicating whether to use cache.

        Raises:
            None
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the SeamlessM4Tv2ForTextToText class.

        This method is used to reorder the past_key_values cache based on the provided beam_idx.
        It returns the reordered cache.

        Args:
            past_key_values (tuple): A tuple containing the past key values.
            beam_idx (torch.Tensor): A tensor representing the indices to reorder the past_key_values.

        Returns:
            tuple: A tuple representing the reordered cache.

        Raises:
            None.

        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class SeamlessM4Tv2ForSpeechToText(SeamlessM4Tv2PreTrainedModel):

    """A class for generating speech-to-text transcriptions using the SeamlessM4Tv2 model.

    This class represents a speech-to-text model based on the SeamlessM4Tv2 architecture.
    It provides methods for initializing the model, getting the encoder and decoder components, setting and getting the
    output and input embeddings, tying weights, constructing the model for training or inference, and generating
    transcriptions.

    Attributes:
        shared (nn.Embedding): The shared embedding layer for the model.
        speech_encoder (SeamlessM4Tv2SpeechEncoder): The speech encoder component of the model.
        text_decoder (SeamlessM4Tv2Decoder): The text decoder component of the model.
        lm_head (nn.Dense): The linear layer for projecting decoder outputs to the vocabulary size.

    Methods:
        __init__: Initializes the model with the given configuration.
        get_encoder: Returns the speech encoder component of the model.
        get_decoder: Returns the text decoder component of the model.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets the input embeddings of the model.
        _tie_weights: Ties the word embeddings of the text decoder and the shared embedding layer if configured.
        construct: Constructs the model for training or inference.
        generate: Generates sequences of token ids.
        prepare_inputs_for_generation: Prepares the inputs for generation.

    """
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config: SeamlessM4Tv2Config):
        """
        Initializes an instance of the SeamlessM4Tv2ForSpeechToText class.

        Args:
            self: The instance of the class.
            config (SeamlessM4Tv2Config): The configuration object containing various settings.
                It must be an instance of the SeamlessM4Tv2Config class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_encoder
    def get_encoder(self):
        """
        Method to retrieve the speech encoder from the SeamlessM4Tv2ForSpeechToText class.

        Args:
            self: An instance of the SeamlessM4Tv2ForSpeechToText class.
                This parameter is required to access the attributes and methods of the class.

        Returns:
            speech_encode: This method returns the speech encoder associated with the instance of the class.

        Raises:
            None.
        """
        return self.speech_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_decoder
    def get_decoder(self):
        """
        Retrieve the text decoder used for decoding SeamlessM4Tv2 audio data into text for speech-to-text conversion.

        Args:
            self (SeamlessM4Tv2ForSpeechToText): An instance of the SeamlessM4Tv2ForSpeechToText class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings of the SeamlessM4Tv2ForSpeechToText model.

        Args:
            self: An instance of the SeamlessM4Tv2ForSpeechToText class.

        Returns:
            lm_head: This method returns the output embeddings of the SeamlessM4Tv2ForSpeechToText model.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the SeamlessM4Tv2ForSpeechToText model.

        Args:
            self (SeamlessM4Tv2ForSpeechToText): The instance of the SeamlessM4Tv2ForSpeechToText class.
            new_embeddings (any): The new embeddings to be set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_input_embeddings
    def get_input_embeddings(self):
        """
        Returns the input embeddings for the SeamlessM4Tv2ForSpeechToText model.

        Args:
            self: An instance of the SeamlessM4Tv2ForSpeechToText class.

        Returns:
            None. This method does not return any value.

        Raises:
            None.

        This method retrieves the input embeddings from the text decoder of the SeamlessM4Tv2ForSpeechToText model.
        The input embeddings are essential for representing the textual input as numerical vectors.
        These embeddings are used as input to the model's further processing steps, such as encoding and decoding.

        Note:
            The input embeddings are computed based on the tokens embedded by the text decoder.
            The text decoder is an integral part of the SeamlessM4Tv2ForSpeechToText model architecture.
        """
        return self.text_decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2ForSpeechToText model.

        Args:
            self (SeamlessM4Tv2ForSpeechToText): The instance of the SeamlessM4Tv2ForSpeechToText class.
            value (torch.Tensor): The input embeddings to be set for the model.
                This should be a tensor of shape (vocab_size, embed_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.text_decoder.embed_tokens = value

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._tie_weights
    def _tie_weights(self):
        """
        Ties the weights of the text decoder and the language model head if the configuration specifies to do so.

        Args:
            self (SeamlessM4Tv2ForSpeechToText): The current instance of the SeamlessM4Tv2ForSpeechToText class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.forward
    def construct(
        self,
        input_features: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """
        Constructs the SeamlessM4Tv2ForSpeechToText model.

        This method takes the following parameters:

        - self: The instance of the class.
        - input_features (mindspore.Tensor, optional): The input features. Default is None.
        - attention_mask (mindspore.Tensor, optional): The attention mask. Default is None.
        - decoder_input_ids (mindspore.Tensor, optional): The decoder input IDs. Default is None.
        - decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask. Default is None.
        - encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs. Default is None.
        - past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values. Default is None.
        - inputs_embeds (mindspore.Tensor, optional): The input embeddings. Default is None.
        - decoder_inputs_embeds (mindspore.Tensor, optional): The decoder input embeddings. Default is None.
        - labels (mindspore.Tensor, optional): The labels. Default is None.
        - use_cache (bool, optional): Whether to use cache. Default is None.
        - output_attentions (bool, optional): Whether to output attentions. Default is None.
        - output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
        - return_dict (bool, optional): Whether to return a dictionary. Default is None.
        - **kwargs: Additional keyword arguments.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The output of the model.

        Raises:
            None.
        """
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.generate
    def generate(
        self,
        input_features=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`mindspore.Tensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.

            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
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
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`:
                A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.
                The possible [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            inputs = kwargs.get("input_embeds") if input_features is None else input_features
            inputs = (
                inputs
                if inputs is not None
                else kwargs.get("encoder_outputs", {"last_hidden_state": None})["last_hidden_state"]
            )
            batch_size = len(inputs)

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                # also accept __xxx__
                tgt_lang = tgt_lang.replace("__", "")
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in
                        {', '.join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                    )
                # tgt_lang gets priority over decoder input ids
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                text_decoder_input_ids = mindspore.tensor([[text_tgt_lang_id]] * batch_size)
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )
        return super().generate(
            input_features,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepares input for generation of speech-to-text using the SeamlessM4Tv2 model.

        Args:
            self: The class instance.
            decoder_input_ids (torch.Tensor): Input tensor IDs for decoder.
            past_key_values (torch.Tensor, optional): The previous key values. Defaults to None.
            attention_mask (torch.Tensor, optional): Mask to focus on relevant input tokens. Defaults to None.
            use_cache (bool, optional): Flag to use cache. Defaults to None.
            encoder_outputs (tuple, optional): Tuple containing the outputs of the encoder. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing input tensors for the model.

        Raises:
            None
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the given beam index in the SeamlessM4Tv2ForSpeechToText class.

        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
                Each layer's past state consists of:

                - past_state: A tensor representing the past state of shape (batch_size, sequence_length, hidden_size).
                - attention_mask: A tensor representing the attention mask of shape (batch_size, sequence_length).
            beam_idx (int): The index of the beam for reordering the cache.

        Returns:
            None: This method modifies the cache in-place.

        Raises:
            None.

        This method reorders the cache by selecting the past states and attention masks for the given beam index.
        It returns the reordered cache with the past states and attention masks for all layers, excluding the other
        cached values. The cache is modified directly, and the method does not return any value.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class SeamlessM4Tv2ForTextToSpeech(SeamlessM4Tv2PreTrainedModel):

    """
    The `SeamlessM4Tv2ForTextToSpeech` class is a subclass of `SeamlessM4Tv2PreTrainedModel` that represents a model
    for text-to-speech synthesis. It is designed specifically for the SeamlessM4Tv2 architecture.

    This class contains methods for generating translated audio waveforms from input text. It utilizes two sub-models:
    a text model and a speech model. The text model generates intermediate text tokens, which are then passed to the
    speech model for synthesis.

    Methods:
        `__init__`: Initializes the `SeamlessM4Tv2ForTextToSpeech` class with a given configuration.
        `get_encoder`: Returns the text encoder of the model.
        `get_decoder`: Returns the text decoder of the model.
        `get_output_embeddings`: Returns the output embeddings of the model.
        `set_output_embeddings`: Sets the output embeddings of the model to the given embeddings.
        `get_input_embeddings`: Returns the input embeddings of the model.
        `set_input_embeddings`: Sets the input embeddings of the model to the given value.
        `_tie_weights`: Ties the weights of the model's embedding layers if specified in the configuration.
        `construct`: Constructs the model for text-to-speech synthesis.
        `generate`: Generates translated audio waveforms from input text.
        `prepare_inputs_for_generation`: Prepares the inputs for generation by the model.
        `_reorder_cache`: Reorders the cached states during beam search.

    Please refer to the method docstrings for more detailed information on their functionality and usage.
    """
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config: SeamlessM4Tv2Config):
        """
        Initializes an instance of the SeamlessM4Tv2ForTextToSpeech class.

        Args:
            self: The instance of the class.
            config (SeamlessM4Tv2Config):
                The configuration object that holds various settings for the model.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The dimensionality of the hidden states.
                - pad_token_id (int): The ID of the padding token.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_encoder
    def get_encoder(self):
        """
        Returns the text encoder for the SeamlessM4Tv2ForTextToSpeech class.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToSpeech class.

        Returns:
            text_encoder: returns the text encoder associated with the class.

        Raises:
            None.
        """
        return self.text_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_decoder
    def get_decoder(self):
        """
        Method to retrieve the text decoder used for SeamlessM4Tv2ForTextToSpeech.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToSpeech class.
                This parameter is required for accessing the current instance.

        Returns:
            text_decoder: The method returns the text decoder associated with the SeamlessM4Tv2ForTextToSpeech instance.

        Raises:
            None.
        """
        return self.text_decoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_output_embeddings
    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the SeamlessM4Tv2ForTextToSpeech class.

        Args:
            self (SeamlessM4Tv2ForTextToSpeech): An instance of the SeamlessM4Tv2ForTextToSpeech class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for a SeamlessM4Tv2ForTextToSpeech instance.

        Args:
            self (SeamlessM4Tv2ForTextToSpeech): The instance of SeamlessM4Tv2ForTextToSpeech.
            new_embeddings (object): The new embeddings to be set as output embeddings for the instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.get_input_embeddings
    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the text decoder.

        Args:
            self: The instance of the SeamlessM4Tv2ForTextToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2ForTextToSpeech model.

        Args:
            self (SeamlessM4Tv2ForTextToSpeech): The instance of the SeamlessM4Tv2ForTextToSpeech class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            ValueError: If the input embeddings provided are not of the expected shape or type.
            TypeError: If the input value is not a torch.Tensor object.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech._tie_weights
    def _tie_weights(self):
        """
        Ties the weights of the shared layers in the SeamlessM4Tv2ForTextToSpeech model.

        Args:
            self: An instance of the SeamlessM4Tv2ForTextToSpeech class.

        Returns:
            None

        Raises:
            None
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.forward with SeamlessM4T->SeamlessM4Tv2
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        '''
        This method constructs a text-to-speech model for the SeamlessM4Tv2 architecture.

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor, optional): The input tensor containing the indices of input tokens.
                Default is None.
            attention_mask (Optional[mindspore.Tensor], optional):
                The tensor indicating which tokens should be attended to. Default is None.
            decoder_input_ids (Optional[mindspore.Tensor], optional):
                The input tensor containing the indices of decoder tokens. Default is None.
            decoder_attention_mask (Optional[mindspore.Tensor], optional): The tensor indicating which tokens should be
                attended to in the decoder. Default is None.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]], optional): The outputs of the encoder model.
                Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]], optional): The past key values for the decoder.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor], optional): The embedded representation of inputs. Default is None.
            decoder_inputs_embeds (Optional[mindspore.Tensor], optional): The embedded representation of decoder inputs.
                Default is None.
            labels (Optional[mindspore.Tensor], optional): The tensor containing the labels for the model.
                Default is None.
            use_cache (Optional[bool], optional): Flag indicating whether to use caching. Default is None.
            output_attentions (Optional[bool], optional): Flag indicating whether to output attentions.
                Default is None.
            output_hidden_states (Optional[bool], optional): Flag indicating whether to output hidden states.
                Default is None.
            return_dict (Optional[bool], optional): Flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The output of the model, which can be either a
                Seq2SeqLMOutput object or a tuple containing a mindspore.Tensor.

        Raises:
            None
        '''
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This is the same forward method as `SeamlessM4Tv2ForTextToText`."
                "It doesn't use the text-to-unit model `SeamlessM4Tv2TextToUnitForConditionalGeneration`."
                "If you want to generate speech, use the `.generate` method."
            )
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        encoder_attention_mask = attention_mask

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.
                - This means you can, for example, specify a generation strategy for one generation but not for the
                other.

        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor]]`:

                - If `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
                - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
                sequence_length)`and and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")

        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = mindspore.tensor([[text_tgt_lang_id]] * batch_size)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_ids, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        if attention_mask is not None:
            # repeat attention mask alongside batch dimension
            attention_mask = ops.repeat_interleave(attention_mask, num_return_sequences, axis=0)
        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = ops.repeat_interleave(encoder_hidden_states, num_return_sequences, axis=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = ops.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )

        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = ops.cat([pad_zero, t2u_char_count_per_id, pad_zero], axis=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(axis=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = ops.softmax(t2u_logits, axis=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = ops.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.copy()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        speaker_id = mindspore.tensor([[speaker_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs required for generation in the SeamlessM4Tv2ForTextToSpeech class.

        Args:
            self: The instance of the class.
            decoder_input_ids (Tensor): The input tensor for the decoder, representing the token ids for the input sequence.
                Its shape is [batch_size, sequence_length] where batch_size is the number of input sequences and
                sequence_length is the length of each sequence.
            past_key_values (Tuple, optional): The past key values used for fast decoding. Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor to be applied on the input sequence.
                Its shape is [batch_size, sequence_length] and the values are 0 for padding tokens and 1 for non-padding
                tokens. Defaults to None.
            use_cache (bool, optional): Whether to use caching for fast decoding. Defaults to None.
            encoder_outputs (Tensor, optional): The output tensor from the encoder.
                Its shape is [batch_size, sequence_length, hidden_size] where hidden_size is the size of the hidden
                state of the encoder.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids': None (currently not used)
                - 'encoder_outputs': The encoder outputs tensor
                - 'past_key_values': The past key values for fast decoding
                - 'decoder_input_ids': The modified decoder input ids
                - 'attention_mask': The attention mask tensor
                - 'use_cache': The flag for using caching

        Raises:
            None
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForTextToSpeech._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        """
        This method '_reorder_cache' is defined in the class 'SeamlessM4Tv2ForTextToSpeech' and is used to reorder
        the cache based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key and value tensors for each layer in the model.
                Each element of the tuple is a tuple containing the past key and value tensors for a specific layer.
                The purpose of this parameter is to provide the past key and value tensors for reordering the cache.
                Restrictions: It should be a valid tuple of tensors.
            beam_idx (tensor): A tensor containing the indices of the beams to be used for reordering the cache.
                The purpose of this parameter is to specify the indices of the beams for reordering the cache.
                Restrictions: It should be a valid tensor containing the beam indices.

        Returns:
            None: This method does not return any value. Instead, it updates the 'reordered_past' variable
                and returns None.

        Raises:
            None: However, potential exceptions that may be raised during the execution of this method could include
                IndexError if the beam index is out of  range or TypeError if the input parameters are not of the
                expected types.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class SeamlessM4Tv2ForSpeechToSpeech(SeamlessM4Tv2PreTrainedModel):

    """
    This class is an implementation of the SeamlessM4Tv2 model for speech-to-speech translation.
    It extends the SeamlessM4Tv2PreTrainedModel class and provides methods for generating translated audio waveforms.

    Example:
        ```python
        >>> model = SeamlessM4Tv2ForSpeechToSpeech(config)
        >>> outputs = model(input_features, tgt_lang, speaker_id, **kwargs)
        ```

    Attributes:
        shared (nn.Embedding): Embedding layer for shared tokens.
        speech_encoder (SeamlessM4Tv2SpeechEncoder): Speech encoder module.
        text_decoder (SeamlessM4Tv2Decoder): Text decoder module.
        lm_head (nn.Dense): Dense layer for language modeling.
        t2u_model (SeamlessM4Tv2TextToUnitForConditionalGeneration): Text-to-unit model for conditional generation.
        vocoder (SeamlessM4Tv2CodeHifiGan): Vocoder model for speech synthesis.

    Methods:
        get_encoder(): Returns the speech encoder module.
        get_decoder(): Returns the text decoder module.
        get_output_embeddings(): Returns the output embeddings.
        set_output_embeddings(new_embeddings): Sets the output embeddings to the provided new_embeddings.
        get_input_embeddings(): Returns the input embeddings.
        set_input_embeddings(value): Sets the input embeddings to the provided value.
        _tie_weights(): Ties the weights of the word embeddings and the shared layer if tie_word_embeddings is True.
        construct(): Constructs the model given the input features, attention masks, decoder input ids,
            and other optional parameters.
        generate(): Generates translated audio waveforms given input features, target language, speaker ID,
            and other optional parameters.
        _reorder_cache(): Reorders the cache of past key values based on beam indices.
        prepare_inputs_for_generation(): Prepares the inputs for generation by handling past key values and
            decoder input ids.

    Note:
        This class is designed for speech-to-speech translation using the SeamlessM4Tv2 model.
    """
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_encoder
    def get_encoder(self):
        """
        This method retrieves the speech encoder for the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self: The instance of the SeamlessM4Tv2ForSpeechToSpeech class.

        Returns:
            speech_encoder: This method returns the speech encoder associated with the instance of the class.

        Raises:
            None.
        """
        return self.speech_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_decoder
    def get_decoder(self):
        """
        Method to retrieve the text decoder for SeamlessM4Tv2ForSpeechToSpeech.

        Args:
            self: An instance of the class SeamlessM4Tv2ForSpeechToSpeech. It is required for accessing the text decoder.

        Returns:
            text_decoder: The method returns the text decoder associated with the instance of
                SeamlessM4Tv2ForSpeechToSpeech.

        Raises:
            None.
        """
        return self.text_decoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings of the SeamlessM4Tv2ForSpeechToSpeech model.

        Args:
            self: An instance of the SeamlessM4Tv2ForSpeechToSpeech class.

        Returns:
            lm_head: This method returns the output embeddings of the model, which are used for downstream tasks
                such as speech-to-speech conversion.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self (SeamlessM4Tv2ForSpeechToSpeech): The instance of the SeamlessM4Tv2ForSpeechToSpeech class.
            new_embeddings (object): The new embeddings to be set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.get_input_embeddings
    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self (SeamlessM4Tv2ForSpeechToSpeech): An instance of the SeamlessM4Tv2ForSpeechToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self: The instance of the SeamlessM4Tv2ForSpeechToSpeech class.
            value: The input embeddings to be set for the text decoder.
                It should be of type 'value' that can be assigned to the 'embed_tokens' attribute of the text decoder.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_decoder.embed_tokens = value

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech._tie_weights
    def _tie_weights(self):
        """
        Ties the weights of the text decoder and language model head to the shared embeddings if specified in the
        configuration.

        Args:
            self (SeamlessM4Tv2ForSpeechToSpeech): The instance of the SeamlessM4Tv2ForSpeechToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.forward with SeamlessM4T->SeamlessM4Tv2
    def construct(
        self,
        input_features: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """
        Constructs the SeamlessM4Tv2ForSpeechToSpeech model.

        Args:
            self: The object itself.
            input_features (mindspore.Tensor, optional): The input features. Default: None.
            attention_mask (mindspore.Tensor, optional): The attention mask. Default: None.
            decoder_input_ids (mindspore.Tensor, optional): The decoder input IDs. Default: None.
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask. Default: None.
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs. Default: None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values. Default: None.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings. Default: None.
            decoder_inputs_embeds (mindspore.Tensor, optional): The decoder input embeddings. Default: None.
            labels (mindspore.Tensor, optional): The labels. Default: None.
            use_cache (bool, optional): Whether to use cache. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary. Default: None.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The output of the model.

        Raises:
            None.
        """
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This is the same forward method as `SeamlessM4Tv2ForSpeechToText`. It doesn't use `self.t2u_model`."
                "If you want to generate speech, use the `generate` method."
            )

            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        encoder_attention_mask = attention_mask
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_features: Optional[mindspore.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_features (`mindspore.Tensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.

            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.
                This means you can, for example, specify a generation strategy for one generation but not for the
                other.


        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor]]`:

                - If `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
                - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
                      sequence_length)`and and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_features) if input_features is not None else len(kwargs.get("inputs_embeds"))

        if tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")
        else:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            for key in ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = mindspore.tensor([[text_tgt_lang_id]] * batch_size)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        text_generation_output = super().generate(input_features, **kwargs_text)
        sequences = text_generation_output.sequences

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get last_hidden_state from encoder
        encoder_hidden_states = self.speech_encoder(input_features=input_features, attention_mask=attention_mask)[0]

        # input modality = speech so new attention mask for the decoder
        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
            )

            # repeat attention mask alongside batch dimension
            attention_mask = ops.repeat_interleave(attention_mask, num_return_sequences, axis=0)

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = ops.repeat_interleave(encoder_hidden_states, num_return_sequences, axis=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = ops.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )

        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = ops.cat([pad_zero, t2u_char_count_per_id, pad_zero], axis=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(axis=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = ops.softmax(t2u_logits, axis=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = ops.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.copy()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        speaker_id = mindspore.tensor([[speaker_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    @staticmethod
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        """
        This method '_reorder_cache' is a static method within the class 'SeamlessM4Tv2ForSpeechToSpeech'.
        It reorders the cache based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing past key values from the model.
                It represents the cache to be reordered.
            beam_idx (Tensor): A tensor representing the beam index to use for reordering.
                It specifies the order in which the cache should be rearranged.

        Returns:
            None: This method does not return any value. It modifies the 'past_key_values' in place to reorder the
                cache based on the 'beam_idx'.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToSpeech.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs required for generation in the SeamlessM4Tv2ForSpeechToSpeech class.

        Args:
            self (object): The instance of the class.
            decoder_input_ids (tensor): The input tensor for the decoder. It represents the input tokens for generation.
            past_key_values (tuple, optional): The past key values for autoregressive generation. Defaults to None.
            attention_mask (tensor, optional): The attention mask tensor to be applied to the input. Defaults to None.
            use_cache (bool, optional): Flag to use caching for generation. Defaults to None.
            encoder_outputs (tensor, optional): The output of the encoder. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared inputs for generation including 'input_ids', 'encoder_outputs',
                'past_key_values', 'decoder_input_ids', 'attention_mask', and 'use_cache'.

        Raises:
            None.
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }


class SeamlessM4Tv2Model(SeamlessM4Tv2PreTrainedModel):

    """
    This class represents a model for SeamlessM4Tv2 with various functionalities for text and speech processing.
    It includes methods for setting and getting modalities, generating translations, preparing inputs for generation,
    and more. The model consists of components such as text encoder, speech encoder, text decoder, LM head, text-to-unit
    model for conditional generation, and vocoder. The class provides flexibility in handling different modalities,
    generating translated text and audio waveforms, and managing cache for efficient generation. Additionally, it offers
    methods for tying weights and reordering cache during generation processes.

    The class inherits from SeamlessM4Tv2PreTrainedModel and encompasses a wide range of features and capabilities for
    seamless text and speech processing tasks. It provides a comprehensive and versatile solution for natural language
    processing and speech synthesis applications.
    """
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config, current_modality="text"):
        """
        Initializes the SeamlessM4Tv2Model class.

        Args:
            self: The object itself.
            config (object): The configuration object containing various settings.
            current_modality (str, optional): The current modality being used, default is 'text'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # these models already call post_init in their initialization
        self.t2u_model = SeamlessM4Tv2TextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4Tv2CodeHifiGan(config)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_modality
    def set_modality(self, modality="text"):
        """
        Method to set the modality of the SeamlessM4Tv2Model.

        Args:
            self (SeamlessM4Tv2Model): The instance of the SeamlessM4Tv2Model class.
            modality (str): Specifies the modality to be set. Accepts either 'text' or 'speech'.

        Returns:
            None.

        Raises:
            ValueError: If the provided modality is not 'text' or 'speech'.
        """
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_encoder
    def get_encoder(self):
        """
        Method to retrieve the appropriate encoder based on the current modality in the SeamlessM4Tv2Model class.

        Args:
            self: Instance of the SeamlessM4Tv2Model class.

        Returns:
            text_encoder:
                Returns the text_encoder if the current modality is 'text',
                otherwise returns the speech_encoder. Returns None if no encoder is found.

        Raises:
            None.
        """
        if self.current_modality == "text":
            return self.text_encoder
        return self.speech_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_output_embeddings
    def get_output_embeddings(self):
        """
        This method is defined in the 'SeamlessM4Tv2Model' class and is named 'get_output_embeddings'.
        It takes '1' parameter which is 'self'.

        Args:
            self: An instance of the 'SeamlessM4Tv2Model' class. It represents the current object of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the SeamlessM4Tv2Model.

        Args:
            self (SeamlessM4Tv2Model): The instance of the SeamlessM4Tv2Model class.
                This parameter refers to the current instance of the SeamlessM4Tv2Model object.
            new_embeddings (object): The new output embeddings to be set for the model.
                It can be any valid object that represents the new embeddings to be assigned.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.get_input_embeddings
    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings from the text decoder of the SeamlessM4Tv2Model.

        Args:
            self:
                Represents the instance of the SeamlessM4Tv2Model class.

                - Type: SeamlessM4Tv2Model
                - Purpose: Allows access to the text decoder to retrieve input embeddings.
                - Restrictions: None

        Returns:
            None:

                - Type: None
                - Purpose: The method returns None as it directly returns the embed_tokens from the text decoder.

        Raises:
            None
        """
        return self.text_decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the SeamlessM4Tv2Model.

        Args:
            self (SeamlessM4Tv2Model): The instance of the SeamlessM4Tv2Model.
            value: The input embeddings to be set. This should be a tensor.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel._tie_weights
    def _tie_weights(self):
        """
        Ties the weights of shared embeddings and the language model head if specified in the configuration.

        Args:
            self: SeamlessM4Tv2Model
                The instance of SeamlessM4Tv2Model class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.forward with SeamlessM4T->SeamlessM4Tv2
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        input_features: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """Constructs the SeamlessM4Tv2Model.

        Args:
            self: The object instance.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Default is None.
            input_features (Optional[mindspore.Tensor]): The input features. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask. Default is None.
            decoder_input_ids (Optional[mindspore.Tensor]): The decoder input token IDs. Default is None.
            decoder_attention_mask (Optional[mindspore.Tensor]): The decoder attention mask. Default is None.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]]): The encoder outputs. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs. Default is None.
            decoder_inputs_embeds (Optional[mindspore.Tensor]): The embedded decoder inputs. Default is None.
            labels (Optional[mindspore.Tensor]): The labels. Default is None.
            use_cache (Optional[bool]): Whether to use cache. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The model output.

        Raises:
            ValueError: If `input_ids`, `input_features`, `inputs_embeds`, and `encoder_outputs` are all empty.
            TypeError: If `encoder_outputs` is not an instance of `BaseModelOutput`.
            UserWarning: If `labels` is provided, the `use_cache` argument is changed to `False`.
            UserWarning: If `input_ids` is not `None` but `input_features` has been given. `input_features` will be
                used instead of `input_ids`.
            UserWarning: If `inputs_embeds` is not `None` but `input_features` has been given. `input_features`
                will be used instead of `inputs_embeds`.
            UserWarning: This method calls the same method `forward` as `SeamlessM4Tv2ForTextToText` and
                `SeamlessM4Tv2ForSpeechToText` depending on the input modality. If you want to generate speech, use the
                `generate` method.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if input_ids is None and input_features is None and inputs_embeds is None and encoder_outputs is None:
            raise ValueError(
                "`input_ids`,`input_features`, `inputs_embeds` and `encoder_outputs` are all empty. Make sure at least one of them is not."
            )
        elif input_features is not None:
            if input_ids is not None:
                logger.warning(
                    "`input_ids` is not `None` but `input_features` has been given."
                    "`input_features` will be used in priority through the `speech_encoder`. "
                    "Make sure that `input_features` and `input_ids` are mutually exclusive."
                )

            if inputs_embeds is not None:
                logger.warning(
                    "`inputs_embeds` is not `None` but `input_features` has been given."
                    "`input_features` will be used in priority through `speech_encoder`. "
                    "`inputs_embeds` will be ignored."
                )

            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This calls the same method `forward` as `SeamlessM4Tv2ForTextToText` and `SeamlessM4Tv2ForSpeechToText`"
                "depending on the input modality. If you want to generate speech, use the `generate` method."
            )

            self.set_modality("speech")

            encoder_outputs = self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif input_ids is not None or inputs_embeds is not None:
            # if encoder_outputs is not None, it's probably used within a .generate method so no need to warn
            logger.warning(
                "This calls the same method `forward` as `SeamlessM4Tv2ForTextToText` and `SeamlessM4Tv2ForSpeechToText`"
                "depending on the input modality. If you want to generate speech, use the `generate` method."
            )
            self.set_modality("text")
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
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

        encoder_attention_mask = attention_mask
        # input modality = speech so new attention mask
        if self.current_modality == "speech" and attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=encoder_outputs[0], seq_lens=sub_sampled_lengths
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        input_features: Optional[mindspore.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4Tv2GenerationOutput]:
        """
        Generates translated token ids and/or translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            input_features (`mindspore.Tensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
                ignored.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.

            kwargs (*optional*):
                Remaining dictioy of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.
                - This means you can, for example, specify a generation strategy for one generation but not for the
                other.

        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor], ModelOutput]`:

                - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
                - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
                shape `(batch_size, sequence_length)`and and `waveform_lengths` which gives the length of each sample.
                - If `generate_speech=False`, it will returns `ModelOutput`.
        """
        if input_ids is None and input_features is None and kwargs.get("inputs_embeds", None) is None:
            raise ValueError(
                "`input_ids`,`input_features` and `inputs_embeds` are all empty. Make sure at least one of them is not."
            )

        if generate_speech and tgt_lang is None:
            raise ValueError("You must specify a `tgt_lang` to generate translated speech.")

        if tgt_lang is not None:
            # also accept __xxx__
            tgt_lang = tgt_lang.replace("__", "")
            if generate_speech:
                keys_to_check = ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]
            else:
                keys_to_check = ["text_decoder_lang_to_code_id"]
            for key in keys_to_check:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(
                        f"""This model generation config doesn't have a `{key}` key which maps the target language
                        to the right token id. Make sure to load the right generation config."""
                    )
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model.
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4Tv2 supports
                    more languages for text translation than for speech synthesis."""
                    )

        batch_size = (
            len(input_features)
            if input_features is not None
            else (len(input_ids) if input_ids is not None else len(kwargs.get("inputs_embeds")))
        )

        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text["output_hidden_states"] = True
        kwargs_text["return_dict_in_generate"] = True
        kwargs_text["output_scores"] = True

        text_decoder_input_ids = kwargs_text.get("decoder_input_ids")
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        if tgt_lang is not None:
            # tgt_lang gets priority over decoder input ids
            text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
            text_decoder_input_ids = mindspore.tensor([[text_tgt_lang_id]] * batch_size)

        kwargs_text["decoder_input_ids"] = text_decoder_input_ids

        # first generation
        if input_features is not None:
            self.set_modality("speech")
            if input_ids is not None:
                logger.warning(
                    "`input_features` and `input_ids` are both non empty. `input_features` will be used in priority "
                    "through the speech encoder. Make sure `input_features=None` if you want to use the text encoder."
                )
            text_generation_output = super().generate(input_features=input_features, **kwargs_text)
        else:
            self.set_modality("text")
            text_generation_output = super().generate(input_ids=input_ids, input_features=None, **kwargs_text)
        sequences = text_generation_output.sequences

        if not generate_speech:
            return text_generation_output

        # prepare second generation
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))

        # get encoder last hidden states
        if self.current_modality == "speech":
            # get last_hidden_state from encoder - must do a pass through the speech encoder
            encoder_hidden_states = self.speech_encoder(
                input_features=input_features, attention_mask=attention_mask
            ).last_hidden_state

            # input modality = speech so new attention mask for the decoder
            if attention_mask is not None:
                sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask)
                attention_mask = _compute_new_attention_mask(
                    hidden_states=encoder_hidden_states, seq_lens=sub_sampled_lengths
                )
        else:
            encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        if attention_mask is not None:
            # repeat attention mask alongside batch dimension
            attention_mask = ops.repeat_interleave(attention_mask, num_return_sequences, axis=0)

        # repeat attention mask alongside batch dimension
        encoder_hidden_states = ops.repeat_interleave(encoder_hidden_states, num_return_sequences, axis=0)

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences[:, :-1],  # Manually trim the final EOS token
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences[:, :-1] != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # REMOVE EOS and lang_id
        t2u_input_ids = sequences[:, 2:-1]
        # replace every other EOS
        t2u_input_ids = ops.masked_fill(
            t2u_input_ids, t2u_input_ids == self.generation_config.eos_token_id, pad_token_id
        )

        # compute t2u_char_input_ids
        t2u_subwords = self._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = self._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )
        # Add pads for lang, EOS tokens as per NLLB "source" tokenizer mode.
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = ops.cat([pad_zero, t2u_char_count_per_id, pad_zero], axis=1)
        t2u_char_input_ids = self._get_char_input_ids(
            t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )

        # second pass
        t2u_output = self.t2u_model(
            inputs_embeds=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            **kwargs_speech,
        )

        t2u_logits = t2u_output[0]
        padding_mask = t2u_output[1].bool()

        # The text-to-unit model is non auto-regressive. We keep the ability to use sampling with temperature
        temperature = kwargs_speech.get("temperature", None)
        if (temperature is None or temperature == 1.0) or not kwargs_speech.get("do_sample", False):
            unit_ids = t2u_logits.argmax(axis=-1)
        else:
            t2u_logits = t2u_logits / temperature
            # apply softmax
            probs = ops.softmax(t2u_logits, axis=-1)
            # reshape to 2D: (batch_size, seq_len, t2u_vocab_size) -> (batch_size*seq_len, t2u_vocab_size)
            probs = probs.reshape((-1, probs.shape[2]))
            # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
            unit_ids = ops.multinomial(probs, num_samples=1).view(t2u_logits.shape[0], -1)

        output_unit_ids = unit_ids.copy()

        replace_mask = (unit_ids == self.config.t2u_eos_token_id) | (~padding_mask)
        # replace eos per pad
        unit_ids = unit_ids.masked_fill(replace_mask, self.config.t2u_pad_token_id)

        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        speaker_id = mindspore.tensor([[speaker_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(
            input_ids=unit_ids, speaker_id=speaker_id, lang_id=vocoder_tgt_lang_id
        )

        if return_intermediate_token_ids:
            return SeamlessM4Tv2GenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs for the generation of sequences in the SeamlessM4Tv2Model.

        Args:
            self: The object instance.
            decoder_input_ids (Tensor): The input tensor for the decoder. It represents the input sequence to
                the decoder model.
            past_key_values (tuple, optional): The past key values for the decoder. Default is None.
                It represents the cached key values from previous decoding steps.
            attention_mask (Tensor, optional): The attention mask tensor. It masks the attention mechanism in the model
                and can be used to hide certain elements of the input. Default is None.
            use_cache (bool, optional): Flag to indicate whether to use caching for the decoder. Default is None.
            encoder_outputs (tuple, optional): The output of the encoder model. It represents the output of the
                encoder model that can be used as input to the decoder. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (None): Placeholder for input ids. Not used in the current implementation.
                - 'encoder_outputs' (Tensor): The encoder outputs to be used as input to the decoder.
                - 'past_key_values' (tuple): The cached key values from previous decoding steps.
                - 'decoder_input_ids' (Tensor): The input tensor for the decoder.
                - 'attention_mask' (Tensor): The attention mask tensor for masking the input.
                - 'use_cache' (bool): Flag indicating whether to use caching for the decoder.
        
        Raises:
            None.
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TModel._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache based on the provided beam index.
        
        Args:
            past_key_values (tuple): A tuple containing past key values for each layer.
            beam_idx (Tensor): A tensor representing the beam index.
        
        Returns:
            None: This method does not return any value, but it modifies the past_key_values in place.
        
        Raises:
            IndexError: If the beam index is out of range or invalid.
            TypeError: If the input past_key_values is not in the expected format.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

__all__ = [
    "SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
    "SeamlessM4Tv2ForTextToSpeech",
    "SeamlessM4Tv2ForSpeechToSpeech",
    "SeamlessM4Tv2ForTextToText",
    "SeamlessM4Tv2ForSpeechToText",
    "SeamlessM4Tv2Model",
    "SeamlessM4Tv2PreTrainedModel",
]
