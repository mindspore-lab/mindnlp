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
""" MindSpore SeamlessM4T model."""
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter
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
from .configuration_seamless_m4t import SeamlessM4TConfig


logger = logging.get_logger(__name__)


SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hf-seamless-m4t-medium",
    # See all SeamlessM4T models at https://hf-mirror.com/models?filter=seamless_m4t
]

SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = {
    "microsoft/speecht5_hifigan": "https://hf-mirror.com/microsoft/speecht5_hifigan/resolve/main/config.json",
}


@dataclass
class SeamlessM4TGenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4TModel`], [`SeamlessM4TForTextToText`],
    [`SeamlessM4TForTextToSpeech`], [`SeamlessM4TForSpeechToSpeech`] and [`SeamlessM4TForTextToSpeech`].

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

    indices = ops.arange(mask_seq_len).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4T models that generate speech, attribute kwargs to either the text generation or the
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


############ SPEECH ENCODER related code ################


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SeamlessM4TConformer, feat_extract_activation->speech_encoder_hidden_act
class SeamlessM4TConformerPositionalConvEmbedding(nn.Cell):

    """
    A Python class representing a SeamlessM4TConformerPositionalConvEmbedding, which is used for positional
    convolutional embedding within a Conformer neural network model.
    This class inherits from nn.Cell and includes functionality for applying convolution operations with specific
    configurations for padding and grouping.

    Attributes:
        conv: nn.Conv1d
            A 1D convolutional layer with configurable kernel size, padding, and group settings.

        padding: SeamlessM4TConformerSamePadLayer
            A layer for applying padding to the convolutional output based on specified parameters.

        activation: function
            Activation function to be applied to the output of the convolutional layer.

    Methods:
        __init__:
            Constructor method for initializing the SeamlessM4TConformerPositionalConvEmbedding instance.

        construct:
            Method to perform the sequence of operations on the input hidden states, including convolution,
            padding, activation, and axis swapping.

    Usage:
        Instantiate an object of SeamlessM4TConformerPositionalConvEmbedding with a configuration object and utilize
        the 'construct' method to process input hidden states.
    """
    def __init__(self, config):
        """
        Initialize the SeamlessM4TConformerPositionalConvEmbedding.

        Args:
            self (object): The instance of the class.
            config (object): Configuration object containing parameters for initializing the positional
                convolutional embedding.

                - hidden_size (int): The size of hidden units.
                - num_conv_pos_embeddings (int): The number of convolutional positional embeddings.
                - num_conv_pos_embedding_groups (int): The number of groups for the convolutional positional embeddings.
                - speech_encoder_hidden_act (str): The activation function for the speech encoder hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            group=config.num_conv_pos_embedding_groups,
        )

        # self.conv = weight_norm(self.conv, name="weight", axis=2)

        self.padding = SeamlessM4TConformerSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]

    def construct(self, hidden_states):
        """
        Constructs the positional convolutional embedding for the SeamlessM4TConformerPositionalConvEmbedding class.

        Args:
            self (SeamlessM4TConformerPositionalConvEmbedding):
                The instance of the SeamlessM4TConformerPositionalConvEmbedding class.
            hidden_states (numpy.ndarray):
                The input hidden states with shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: The method modifies the hidden_states in place.

        Raises:
            ValueError: If the input hidden_states is not a numpy array.
            ValueError: If the input hidden_states does not have the correct shape
                (batch_size, sequence_length, hidden_size).
            TypeError: If the input hidden_states data type is not compatible with numpy array operations.
        """
        hidden_states = hidden_states.swapaxes(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRotaryPositionalEmbedding with Wav2Vec2->SeamlessM4T, num_attention_heads->speech_encoder_attention_heads
class SeamlessM4TConformerRotaryPositionalEmbedding(nn.Cell):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """
    def __init__(self, config):
        """
        __init__(self, config)

        Initialize the SeamlessM4TConformerRotaryPositionalEmbedding instance.

        Args:
            self: The instance of the SeamlessM4TConformerRotaryPositionalEmbedding class.
            config: A configuration object containing the parameters for the rotary positional embedding,
                including hidden_size and speech_encoder_attention_heads. It also includes the rotary_embedding_base
                used for calculating the inverse frequency. It is expected to be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        dim = config.hidden_size // config.speech_encoder_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def construct(self, hidden_states):
        """
        Constructs the rotary positional embeddings for the SeamlessM4TConformerRotaryPositionalEmbedding.

        Args:
            self: The instance of the SeamlessM4TConformerRotaryPositionalEmbedding class.
            hidden_states: A tensor representing the hidden states. It should have the shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            None: The method updates the cached_rotary_positional_embedding attribute of the instance.

        Raises:
            None.
        """
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = ops.arange(sequence_length).type_as(self.inv_freq)
        freqs = ops.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = ops.cat((freqs, freqs), axis=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = ops.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerRelPositionalEmbedding with Wav2Vec2->SeamlessM4T
class SeamlessM4TConformerRelPositionalEmbedding(nn.Cell):
    """Relative positional encoding module."""
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4TConformerRelPositionalEmbedding class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' containing the configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.pe = None
        self.extend_pe(mindspore.tensor(0.0).expand(1, self.max_len)) # pylint: disable=too-many-function-args

    def extend_pe(self, x):
        """
        Extends the positional embeddings of the SeamlessM4TConformerRelPositionalEmbedding class.

        Args:
            self (SeamlessM4TConformerRelPositionalEmbedding):
                An instance of the SeamlessM4TConformerRelPositionalEmbedding class.
            x (Tensor): The input tensor to extend the positional embeddings.

        Returns:
            None: The method modifies the positional embeddings in-place.

        Raises:
            None.

        Description:
            This method extends the positional embeddings of the SeamlessM4TConformerRelPositionalEmbedding class
            based on the shape of the input tensor, 'x'. If the existing positional embeddings (pe) are already larger
            than or equal to twice the width of 'x', no modifications are made. If the data type of the positional
            embeddings is different from 'x', the positional embeddings are converted to the data type of 'x'.

            The method then calculates positive and negative positional encodings based on the shape of 'x'.
            The positional encodings are calculated using sine and cosine functions with a positional encoding
            matrix. The calculated positional encodings are flipped and concatenated to form the final positional
            embeddings, which are then assigned to the 'pe' attribute of the SeamlessM4TConformerRelPositionalEmbedding
            instance.
        """
        # Reset the positional encodings
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.shape[1] >= x.shape[1] * 2 - 1:
                if self.pe.dtype != x.dtype:
                    self.pe = self.pe.to(dtype=x.dtype)
                return
        # Suppose `i` is the position of query vector and `j` is the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = ops.zeros(x.shape[1], self.d_model)
        pe_negative = ops.zeros(x.shape[1], self.d_model)
        position = ops.arange(0, x.shape[1], dtype=mindspore.float32).unsqueeze(1)
        div_term = ops.exp(
            ops.arange(0, self.d_model, 2, dtype=mindspore.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = ops.sin(position * div_term)
        pe_positive[:, 1::2] = ops.cos(position * div_term)
        pe_negative[:, 0::2] = ops.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = ops.cos(-1 * position * div_term)

        # Reverse the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = ops.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = ops.cat([pe_positive, pe_negative], axis=1)
        self.pe = pe.to(dtype=x.dtype)

    def construct(self, hidden_states: mindspore.Tensor):
        """
        Constructs the relative positional embeddings for the SeamlessM4TConformer model.

        Args:
            self (SeamlessM4TConformerRelPositionalEmbedding): An instance of the
                SeamlessM4TConformerRelPositionalEmbedding class.
            hidden_states (mindspore.Tensor): The hidden states of the model.

        Returns:
            mindspore.Tensor: The relative position embeddings for the given hidden states.

        Raises:
            None.

        Description:
            This method takes the hidden states of the model and constructs the relative position embeddings.
            It first extends the positional encodings (pe) using the extend_pe() method. Then, it calculates the
            start and end indices for selecting the relevant portion of the positional encodings based on the length
            of the hidden states. Finally, it returns the relative position embeddings for the given hidden states.

            The positional encodings are extended to ensure that there are sufficient embeddings to cover the entire
            sequence of hidden states. The start and end indices are calculated to select the relevant
            portion of the positional encodings that corresponds to the hidden states. This ensures that the relative
            position embeddings are aligned with the hidden states.

        Note:
            The relative position embeddings are used to capture the positional information between different elements
            in the hidden states. They help the model understand the relative positions of tokens in the input sequence,
            which is important for tasks such as machine translation.

        Example:
            ```python
            >>> rel_pos_emb = SeamlessM4TConformerRelPositionalEmbedding()
            >>> hidden_states = mindspore.Tensor(...)
            >>> relative_position_embeddings = rel_pos_emb.construct(hidden_states)
            ```
        """
        self.extend_pe(hidden_states)
        start_idx = self.pe.shape[1] // 2 - hidden_states.shape[1] + 1
        end_idx = self.pe.shape[1] // 2 + hidden_states.shape[1]
        relative_position_embeddings = self.pe[:, start_idx:end_idx]

        return relative_position_embeddings


# Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSamePadLayer with Wav2Vec2->SeamlessM4T
class SeamlessM4TConformerSamePadLayer(nn.Cell):

    """
    This class represents a seamless M4T Conformer layer with same padding.

    Inherits from nn.Cell.

    Attributes:
        num_pad_remove (int): The number of padding elements to remove from the input sequence.

    Methods:
        __init__: Initializes the SeamlessM4TConformerSamePadLayer instance.
        construct: Constructs the hidden states of the SeamlessM4TConformerSamePadLayer.

    """
    def __init__(self, num_conv_pos_embeddings):
        """
        Initializes an instance of the SeamlessM4TConformerSamePadLayer class.

        Args:
            self (SeamlessM4TConformerSamePadLayer): The current object instance.
            num_conv_pos_embeddings (int): The number of convolutional position embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def construct(self, hidden_states):
        """
        Constructs the hidden states by removing padding from the input tensor.

        Args:
            self (SeamlessM4TConformerSamePadLayer): An instance of the SeamlessM4TConformerSamePadLayer class.
            hidden_states (torch.Tensor):
                The input tensor containing hidden states.

                - Shape: (batch_size, sequence_length, hidden_size).
                - Purpose: Represents the hidden states to be processed.
                - Restrictions: None.

        Returns:
            None: The hidden states tensor with padding removed is modified in-place.

        Raises:
            None.
        """
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class SeamlessM4TConformerFeatureProjection(nn.Cell):

    """
    This class represents a feature projection module for the SeamlessM4TConformer model.
    It inherits from the nn.Cell class.

    The feature projection module consists of a layer normalization, a dense projection layer,
    and a dropout layer. It takes in hidden states as input and applies layer normalization,
    followed by a projection and dropout operation. The resulting hidden states are returned.

    Attributes:
        layer_norm (nn.LayerNorm): A layer normalization module that normalizes the input hidden states.
        projection (nn.Dense): A dense projection layer that projects the normalized hidden states.
        dropout (nn.Dropout): A dropout layer that applies dropout to the projected hidden states.

    Methods:
        construct(hidden_states):
            Applies the feature projection to the input hidden states.

            Args:

            - hidden_states (Tensor): Input hidden states to be projected.

            Returns:

           - Tensor: The projected hidden states after applying layer normalization, projection, and dropout.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the SeamlessM4TConformerFeatureProjection class.

        Args:
            self (SeamlessM4TConformerFeatureProjection): The current instance of the class.
            config:
                The configuration parameters for the feature projection.

                - feature_projection_input_dim (int): The input dimension for the feature projection.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_size (int): The hidden size for the projection layer.
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
        """
        Method to construct the feature projection in the SeamlessM4TConformerFeatureProjection class.

        Args:
            self (SeamlessM4TConformerFeatureProjection): The instance of the SeamlessM4TConformerFeatureProjection
                class.
            hidden_states (Tensor): The input hidden states to be processed. Expected to be a tensor.

        Returns:
            None: This method does not return any value directly. The hidden_states are processed and modified in-place.

        Raises:
            None.
        """
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SeamlessM4TConformerFeedForward(nn.Cell):

    """
    The SeamlessM4TConformerFeedForward class represents a feed-forward neural network module for the
    SeamlessM4TConformer model. It inherits from the nn.Cell class and contains methods for initializing the network
    and constructing the feed-forward operations.

    Attributes:
        config: The configuration parameters for the feed-forward network.
        act_fn: The activation function to be used in the network.
        dropout: The dropout probability for the network.

    Methods:
        __init__:
            Initializes the SeamlessM4TConformerFeedForward module with the given configuration, activation function,
            and dropout probability.

        construct:
            Constructs the feed-forward operations on the given hidden states, applying intermediate dense layers,
            activation functions, and dropout. Returns the processed hidden states.
    """
    def __init__(self, config, act_fn=None, dropout=None):
        """
        Initializes the SeamlessM4TConformerFeedForward class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration settings.

                - Type: object
                - Purpose: Holds various configuration parameters for the method.
                - Restrictions: Must be provided as an argument.
            act_fn:
                Activation function to be used.

                - Type: str or callable, optional
                - Purpose: Specifies the activation function to apply.
                - Restrictions: If str, it must be a valid key in the ACT2FN dictionary.
            dropout:
                Dropout rate to be applied.

                - Type: float, optional
                - Purpose: Controls the dropout rate for regularization.
                - Restrictions: Must be a float between 0 and 1. If not provided, config.speech_encoder_dropout is used.

        Returns:
            None.

        Raises:
            None.
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
        Constructs the feed forward layer for the SeamlessM4TConformerFeedForward class.

        Args:
            self (SeamlessM4TConformerFeedForward): The instance of the SeamlessM4TConformerFeedForward class.
            hidden_states (tensor): The input hidden states to be processed by the feed forward layer.

        Returns:
            None.

        Raises:
            TypeError: If the input hidden_states is not a valid tensor.
            ValueError: If the input hidden_states is empty or has invalid shape.
            RuntimeError: If there is an issue during the feed forward layer construction process.
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4TConformerConvolutionModule(nn.Cell):
    """Convolution block used in the conformer block"""
    def __init__(self, config):
        """
        Initializes the SeamlessM4TConformerConvolutionModule.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration parameters for the module.

                - conv_depthwise_kernel_size (int): The kernel size for depthwise convolution.
                - hidden_size (int): The size of the hidden layer.
                - speech_encoder_hidden_act (str): The activation function for the hidden layer.
                - speech_encoder_dropout (float): The dropout rate.

        Returns:
            None.

        Raises:
            ValueError: Raised if the 'config.conv_depthwise_kernel_size' is not an odd number, which is required
                for 'SAME' padding.
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
            padding=0,
            has_bias=False,
        )
        self.glu = nn.GLU(axis=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            pad_mode="same",
            group=config.hidden_size,
            has_bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False,
        )
        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

    def construct(self, hidden_states, attention_mask=None):
        """
        Constructs the SeamlessM4TConformerConvolutionModule.

        Args:
            self: The instance of the SeamlessM4TConformerConvolutionModule class.
            hidden_states (torch.Tensor): The input hidden states. It should have shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): An optional attention mask. It should have the same shape as
                hidden_states. Each element of the mask should be 0 or 1, indicating whether a token is valid or masked.
                If provided, the hidden states corresponding to the masked tokens will be set to 0.0. Default is None.

        Returns:
            torch.Tensor: The transformed hidden states after passing through the SeamlessM4TConformerConvolutionModule.
                It has the same shape as the input hidden states.

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

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


class SeamlessM4TConformerSelfAttention(nn.Cell):
    """Construct a SeamlessM4TConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """
    def __init__(self, config, use_position_embeddings=True):
        """
        Initializes a new instance of the SeamlessM4TConformerSelfAttention class.

        Args:
            self: The object instance.
            config (Config): The configuration object.
            use_position_embeddings (bool, optional): Whether to use position embeddings. Default is True.

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

        if self.position_embeddings_type == "relative":
            # linear transformation for positional encoding
            self.linear_pos = nn.Dense(config.hidden_size, config.hidden_size, has_bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = Parameter(ops.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = Parameter(ops.zeros(self.num_heads, self.head_size))

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention.forward
    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        relative_position_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the self-attention mechanism in the SeamlessM4TConformerSelfAttention class.

        Args:
            self (SeamlessM4TConformerSelfAttention): An instance of the SeamlessM4TConformerSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional attention mask tensor of shape
                (batch_size, sequence_length, sequence_length), where each value is either 0 or 1. It is used to mask
                positions in the attention scores that should be ignored.
            relative_position_embeddings (Optional[mindspore.Tensor]): An optional tensor of shape
                (sequence_length, sequence_length, hidden_size) used for relative position embeddings. Required when
                self.position_embeddings_type is 'rotary' or 'relative'.
            output_attentions (bool): A flag indicating whether to output attention probabilities. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing:

                - hidden_states (mindspore.Tensor): The output hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
                - probs (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length) containing the attention probabilities.
                - None (Optional[Tuple[mindspore.Tensor]]): An optional tuple of attention weights tensors, each of
                shape (batch_size, num_heads, sequence_length, sequence_length). Only returned when output_attentions
                is True.

        Raises:
            ValueError: If self.position_embeddings_type is 'rotary' but relative_position_embeddings is not defined.
            ValueError: If self.position_embeddings_type is 'relative' but relative_position_embeddings is not defined.
        """
        # self-attention mechanism
        batch_size, _, _ = hidden_states.shape

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.swapaxes(1, 2)
        key = key.swapaxes(1, 2)
        value = value.swapaxes(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = ops.matmul(query, key.swapaxes(-2, -1)) / math.sqrt(self.head_size)

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores + attention_mask

        # => (batch, head, time1, time2)
        probs = ops.softmax(scores, axis=-1)
        probs = self.dropout(probs)

        # => (batch, head, time1, d_k)
        hidden_states = ops.matmul(probs, value)

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, probs

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_rotary_embedding
    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        """
        Apply rotary embedding to the hidden states in the SeamlessM4TConformerSelfAttention class.

        Args:
            self: Reference to the instance of the class.
            hidden_states (torch.Tensor): A 3D tensor of shape (batch_size, sequence_length, _) representing the
                input hidden states.
            relative_position_embeddings (torch.Tensor): A 3D tensor of shape (2, sequence_length, ...) containing the
                relative position embeddings.

        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, sequence_length, self.num_heads * self.head_size)
                representing the modified hidden states after applying rotary embedding.

        Raises:
            None
        """
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.swapaxes(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = ops.cat((-rotated_states_end, rotated_states_begin), axis=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.swapaxes(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerSelfAttention._apply_relative_embeddings
    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        """Apply relative embeddings to the given query and key.

        This method applies relative position embeddings to the query and key tensors in the
        SeamlessM4TConformerSelfAttention class.

        Args:
            self (SeamlessM4TConformerSelfAttention): The instance of the SeamlessM4TConformerSelfAttention class.
            query (Tensor): The query tensor.
            key (Tensor): The key tensor.
            relative_position_embeddings (Tensor): The tensor containing relative position embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        # 1. project positional embeddings
        # => (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.shape[0], -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.swapaxes(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.swapaxes(2, 3)

        # 2. Add bias to query
        # => (batch, head, time1, d_k)
        query = query.swapaxes(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).swapaxes(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).swapaxes(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # => (batch, head, time1, time2)
        scores_ac = ops.matmul(q_with_bias_u, key.swapaxes(-2, -1))

        # 4. then compute matrix b and matrix d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = ops.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        zero_pad = ops.zeros((*scores_bd.shape[:3], 1), dtype=scores_bd.dtype)
        scores_bd_padded = ops.cat([zero_pad, scores_bd], axis=-1)
        scores_bd_padded_shape = scores_bd.shape[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.shape[-1] // 2 + 1]

        # 6. sum matrices
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class SeamlessM4TConformerEncoderLayer(nn.Cell):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""
    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4T, attention_dropout->speech_encoder_dropout, ops.nn->nn
    def __init__(self, config):
        """
        Initializes a SeamlessM4TConformerEncoderLayer object.

        Args:
            self (SeamlessM4TConformerEncoderLayer): The instance of the class itself.
            config (object): A configuration object containing parameters for the encoder layer.
                It must have the following attributes:

                - hidden_size (int): The dimension of the hidden layers.
                - speech_encoder_dropout (float): The dropout probability for the speech encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn1 = SeamlessM4TConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm([embed_dim])
        self.self_attn_dropout = nn.Dropout(p=dropout)
        self.self_attn = SeamlessM4TConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4TConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn2 = SeamlessM4TConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm([embed_dim])

    def construct(
        self,
        hidden_states,
        attention_mask: Optional[mindspore.Tensor] = None,
        relative_position_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[mindspore.Tensor] = None,
    ):
        """
        The 'construct' method in the 'SeamlessM4TConformerEncoderLayer' class constructs the encoder layer of a
        Conformer model.

        Args:
            self: Reference to the current instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states for the encoder layer.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask.
                Default is None.
            relative_position_embeddings (Optional[mindspore.Tensor]): Optional tensor for relative position embeddings.
                Default is None.
            output_attentions (bool): A flag indicating whether to output attention weights. Default is False.
            conv_attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the convolution attention
                mask. Default is None.

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor]: The constructed hidden states after processing through the
                encoder layer, along with the attention weights.

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
        hidden_states, attn_weigts = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
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

        return hidden_states, attn_weigts


class SeamlessM4TConformerEncoder(nn.Cell):

    """
    This class represents a SeamlessM4TConformerEncoder which is responsible for encoding input sequences using a
    Conformer model architecture.
    The encoder consists of multiple ConformerEncoderLayer instances stacked on top of each other.
    It handles positional embeddings, dropout, layer normalization, and gradient checkpointing.

    Parameters:
        config: An object containing configuration parameters for the encoder.
        Inherits from: nn.Cell

    Methods:
        __init__: Initializes the SeamlessM4TConformerEncoder with the provided configuration. Sets up positional
            embeddings based on the specified type, dropout, encoder layers, layer normalization, and
            gradient checkpointing.
        construct: Constructs the encoder by processing the input hidden states through each encoder layer.
            It applies dropout, handles attention masks, and computes relative position embeddings.
            Returns the encoded hidden states, hidden states history if enabled, and attention weights if requested.

    Attributes:
        config: Configuration parameters for the encoder.
        embed_positions: Positional embedding module based on the specified type ('relative' or 'rotary').
        dropout: Dropout module for regularization.
        layers: List of ConformerEncoderLayer instances representing the stacked encoder layers.
        layer_norm: Layer normalization module to normalize the hidden states.
        gradient_checkpointing: Flag indicating whether gradient checkpointing is enabled.

    For detailed usage instructions and examples, refer to the official documentation.
    """
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4TConformerEncoder class.

        Args:
            self: An instance of the SeamlessM4TConformerEncoder class.
            config: An object of type Config that contains configuration parameters
                for the SeamlessM4TConformerEncoder.

        Returns:
            None

        Raises:
            None

        This method initializes the SeamlessM4TConformerEncoder with the given configuration parameters.
        It sets the configuration parameters for the instance and initializes the positional embedding based
        on the type of position embedding specified in the configuration. The method also sets the dropout probability,
        creates a list of encoder layers based on the number of layers specified in the configuration, normalizes the
        outputs of the encoder layer using LayerNorm, and sets the gradient checkpointing flag to False.
        """
        super().__init__()
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = SeamlessM4TConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = SeamlessM4TConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)
        self.layers = nn.CellList(
            [SeamlessM4TConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Construct method in the SeamlessM4TConformerEncoder class.

        Args:
            self: The instance of the SeamlessM4TConformerEncoder class.
            hidden_states (tensor): The input hidden states to be processed by the encoder.
            attention_mask (tensor, optional): A tensor representing the attention mask to be applied during processing.
                Defaults to None.
            output_attentions (bool, optional): A flag indicating whether to output the attention weights.
                Defaults to False.
            output_hidden_states (bool, optional): A flag indicating whether to output the hidden states of each layer.
                Defaults to False.
            return_dict (bool, optional): A flag indicating whether to return the outputs as a dictionary.
                Defaults to True.

        Returns:
            None: The method does not explicitly return a value, but updates hidden_states, all_hidden_states,
                and all_self_attentions within the class instance.

        Raises:
            TypeError: If the input arguments are of incorrect types.
            ValueError: If the input hidden_states and attention_mask have incompatible shapes.
            RuntimeError: If an error occurs during processing or if the input tensors are not well-formed.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * float(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).min)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        for _, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = ops.rand([])

            skip_the_layer = bool(self.training and (dropout_probability < self.config.speech_encoder_layerdrop))
            if not skip_the_layer:
                # under deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    relative_position_embeddings=relative_position_embeddings,
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


class SeamlessM4TConformerAdapterLayer(nn.Cell):

    """
    The `SeamlessM4TConformerAdapterLayer` class is a Python class that represents a layer in the SeamlessM4TConformer
    adapter model. This layer is used to adapt the input hidden states using self-attention and feed-forward networks.

    This class inherits from the `nn.Cell` class.

    Attributes:
        `kernel_size` (int): The size of the kernel used in the convolutional layers.
        `stride` (int): The stride used in the convolutional layers.
        `residual_layer_norm` (nn.LayerNorm): A layer normalization module applied to the residual hidden states.
        `residual_conv` (nn.Conv1d): A 1D convolutional layer used to transform the residual hidden states.
        `activation` (nn.GLU): The activation function applied to the transformed residual hidden states.
        `self_attn_layer_norm` (nn.LayerNorm): A layer normalization module applied to the self-attention hidden states.
        `self_attn_conv` (nn.Conv1d): A 1D convolutional layer used to transform the self-attention hidden states.
        `self_attn` (SeamlessM4TConformerSelfAttention): The self-attention module used to compute attention weights.
        `self_attn_dropout` (nn.Dropout): A dropout layer applied to the self-attention hidden states.
        `ffn_layer_norm` (nn.LayerNorm): A layer normalization module applied to the feed-forward hidden states.
        `ffn` (SeamlessM4TConformerFeedForward): The feed-forward module used to transform the feed-forward hidden states.

    Methods:
        `_compute_sub_sample_lengths_from_attention_mask`: Computes the sub-sampled lengths of the hidden states
            based on the attention mask.
        `construct`: Constructs the output hidden states by applying the adapter layer transformations to the
            input hidden states.

    Note:
        This class assumes the existence of the following helper functions: `_compute_new_attention_mask`,
        `_prepare_4d_attention_mask`.

    """
    def __init__(self, config):
        """Initializes an instance of the SeamlessM4TConformerAdapterLayer class.

        Args:
            self: The instance of the class.
            config:
                An object of the configuration class containing the following attributes:

                - hidden_size: An integer representing the size of the hidden dimension.
                - adaptor_dropout: A float representing the dropout probability for adapter layers.
                - adaptor_kernel_size: An integer representing the kernel size for the convolutional layer.
                - adaptor_stride: An integer representing the stride for the convolutional layer.

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
        self.self_attn = SeamlessM4TConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(p=dropout)

        # Feed-forward
        self.ffn_layer_norm = nn.LayerNorm([embed_dim])
        self.ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        """
        Computes the lengths of sub-samples from the given attention mask.

        Args:
            self (SeamlessM4TConformerAdapterLayer): The instance of the SeamlessM4TConformerAdapterLayer class.
            attention_mask (mindspore.Tensor): The attention mask tensor of shape [batch_size, sequence_length].
                It masks the input sequence to exclude certain positions from being attended to.
                The values should be either 0 or 1, where 0 indicates that the position is masked and 1 indicates
                that the position is not masked.

        Returns:
            None.

        Raises:
            None.

        This method calculates the lengths of sub-samples based on the attention mask provided.
        It applies the following steps:

        - Calculate the padding value based on the kernel size.
        - Calculate the sequence lengths by subtracting the sum of all non-masked positions (indicated by 1 in the mask)
        from the total sequence length.
        - Adjust the sequence lengths by considering the padding and kernel size, and divide it by the stride length.
        - Add 1 to the adjusted sequence lengths.
        - Convert the sequence lengths to the float32 data type.
        - Round down the sequence lengths to the nearest integer.

        Note:
            - The padding value is determined by dividing the kernel size by 2 and taking the integer division.
            - The stride length is assumed to be a pre-defined value.
            - The method assumes that the attention mask is a binary tensor with values 0 and 1.

        Example:
            ```python
            >>> # Create an instance of SeamlessM4TConformerAdapterLayer
            >>> adapter_layer = SeamlessM4TConformerAdapterLayer()
            ...
            >>> # Create an attention mask tensor
            >>> attention_mask = mindspore.Tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
            ...
            >>> # Compute the sub-sample lengths from the attention mask
            >>> adapter_layer._compute_sub_sample_lengths_from_attention_mask(attention_mask)
            ```
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
        Constructs a SeamlessM4TConformerAdapterLayer.

        This method applies the necessary transformations and computations to the input `hidden_states` to produce
        the final output `hidden_states`.

        Args:
            self (SeamlessM4TConformerAdapterLayer): The instance of the SeamlessM4TConformerAdapterLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor. It should have a shape of
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask.
                It should have a shape of (batch_size, sequence_length).
            output_attentions (bool): A flag indicating whether to output attentions. Defaults to False.

        Returns:
            mindspore.Tensor: The output hidden states tensor. It has the same shape as the input `hidden_states`.

        Raises:
            None.
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


class SeamlessM4TConformerAdapter(nn.Cell):

    """
    This class represents a seamless multi-task (M4T) Conformer adapter, designed for adapting transformer-based models
    for multi-task learning. The adapter consists of multiple adapter layers that can be stacked on top of each other
    to adapt the model's hidden states for different tasks.

    Attributes:
        layers (nn.CellList): A list of SeamlessM4TConformerAdapterLayer instances, each representing an adapter layer
            in the adapter stack.

    Methods:
        __init__:
            Initializes the SeamlessM4TConformerAdapter instance with the specified configuration.

            Args:

            - config (dict): A dictionary containing configuration parameters for the adapter.

        construct:
            Constructs the adapter by applying each adapter layer in the stack to the input hidden states.

            Args:

            - hidden_states (Tensor): The input hidden states to be adapted by the adapter.
            - attention_mask (Tensor): The attention mask to be applied during adaptation.

            Returns:

            - Tensor: The adapted hidden states after passing through all adapter layers.
    """
    def __init__(self, config):
        """
        Initializes an instance of the SeamlessM4TConformerAdapter class.

        Args:
            self (SeamlessM4TConformerAdapter): The instance of the class itself.
            config:
                A configuration object containing the necessary parameters for initializing the adapter.

                - num_adapter_layers (int): The number of adapter layers to create.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.layers = nn.CellList([SeamlessM4TConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)])

    def construct(self, hidden_states, attention_mask):
        """
        Constructs the SeamlessM4TConformerAdapter by applying the layers to the input hidden states.

        Args:
            self (SeamlessM4TConformerAdapter): An instance of the SeamlessM4TConformerAdapter class.
            hidden_states (Tensor): The input hidden states. It should have a shape of
                [batch_size, sequence_length, hidden_size].
            attention_mask (Tensor): The attention mask tensor. It should have a shape of [batch_size, sequence_length]
                and is used to mask certain positions in the input sequence.

        Returns:
            None.

        Raises:
            None.
        """
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding
class SeamlessM4TSinusoidalPositionalEmbedding(nn.Cell):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Initializes an instance of the SeamlessM4TSinusoidalPositionalEmbedding class.

        Args:
            self: The instance of the class.
            num_positions (int): The number of positions to be considered for the sinusoidal embedding.
                It should be a positive integer.
            embedding_dim (int): The dimension of the embedding vectors. It should be a positive integer.
            padding_idx (Optional[int], optional): The index used for padding.
                If provided, it should be a non-negative integer. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Generate embedding weights for a SeamlessM4TSinusoidalPositionalEmbedding instance.

        Args:
            self (SeamlessM4TSinusoidalPositionalEmbedding): The instance of the
                SeamlessM4TSinusoidalPositionalEmbedding class.
            num_embeddings (int): The number of embeddings to generate.
            embedding_dim (int): The dimensionality of each embedding.
            padding_idx (int, optional): An optional index representing padding. Defaults to None.

        Returns:
            None: This method modifies the weights attribute of the SeamlessM4TSinusoidalPositionalEmbedding instance.

        Raises:
            None.

        This method generates embedding weights for the SeamlessM4TSinusoidalPositionalEmbedding instance by calling
        the get_embedding method. If the instance already has a weights attribute, the dtype of the generated weights
        is converted to match the existing weights. Finally, the generated weights are assigned to the weights attribute
        of the instance.
        """
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype) # pylint: disable=access-member-before-definition
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
        '''
        Construct the sinusoidal positional embedding for input tokens.

        Args:
            self (SeamlessM4TSinusoidalPositionalEmbedding):
                The instance of the SeamlessM4TSinusoidalPositionalEmbedding class.
            input_ids (mindspore.Tensor, optional): The input token IDs. Default is None.
                If provided, the shape should be (batch_size, sequence_length).
            inputs_embeds (mindspore.Tensor, optional): The input embeddings. Default is None.
                If provided, the shape should be (batch_size, sequence_length, embedding_dim).
            past_key_values_length (int, optional): The length of past key values. Default is 0.

        Returns:
            None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None.
            ValueError: If the max_pos exceeds the shape of the weights tensor.

        '''
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
            inputs_embeds (mindspore.Tensor):

        Returns:
            mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64)
        return position_ids.unsqueeze(0).expand(input_shape)+ past_key_values_length


class SeamlessM4TAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Copied from transformers.models.bart.modeling_bart.BartAttention.__init__ with Bart->SeamlessM4T
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4TConfig] = None,
    ):
        """
        Initialize the SeamlessM4TAttention class.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Flag indicating if the attention is used in a decoder context.
                Defaults to False.
            bias (bool): Flag indicating whether to include bias in linear transformations.
            is_causal (bool): Flag indicating if the attention is causal.
            config (Optional[SeamlessM4TConfig]): An optional configuration object for the attention mechanism.

        Returns:
            None.

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

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' is defined within the class 'SeamlessM4TAttention' and is used to reshape the input tensor
        based on the provided sequence length and batch size.

        Args:
            self: An instance of the 'SeamlessM4TAttention' class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None: This method does not return any value. It modifies the input tensor in place to reshape it as per the
                specified sequence length and batch size.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if encoder_hidden_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = encoder_hidden_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == encoder_hidden_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `encoder_hidden_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == encoder_hidden_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(encoder_hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(encoder_hidden_states), -1, bsz)
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


# Copied from transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeDenseActDense with NllbMoe->SeamlessM4T,DenseActDense->FeedForwardNetwork, d_model->hidden_size
class SeamlessM4TFeedForwardNetwork(nn.Cell):

    """
    The SeamlessM4TFeedForwardNetwork class represents a feedforward network for the SeamlessM4T model.
    It inherits from the nn.Cell class and is designed to handle feedforward operations for the SeamlessM4T model.

    Attributes:
        config (SeamlessM4TConfig): An instance of SeamlessM4TConfig class containing configuration parameters.
        ffn_dim (int): The dimension of the feedforward network.

    Methods:
        __init__:
            Initializes the SeamlessM4TFeedForwardNetwork with the given configuration and feedforward network dimension.

        construct:
            Constructs the feedforward network using the provided hidden states.

    Returns:
        The constructed feedforward network for the SeamlessM4T model.
    """
    def __init__(self, config: SeamlessM4TConfig, ffn_dim: int):
        '''
        Initializes the SeamlessM4TFeedForwardNetwork class.

        Args:
            self: The instance of the class.
            config (SeamlessM4TConfig): An instance of the SeamlessM4TConfig class containing the configuration
                parameters for the feed forward network.
            ffn_dim (int): An integer representing the dimensionality of the feed forward network.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type SeamlessM4TConfig.
            ValueError: If ffn_dim is not a positive integer.
        '''
        super().__init__()
        self.fc1 = nn.Dense(config.hidden_size, ffn_dim)
        self.fc2 = nn.Dense(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(p=config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def construct(self, hidden_states):
        """
        Constructs the forward pass of the SeamlessM4TFeedForwardNetwork.

        Args:
            self (SeamlessM4TFeedForwardNetwork): The instance of the SeamlessM4TFeedForwardNetwork class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.

        Returns:
            mindspore.Tensor: The processed hidden states after passing through the network.

        Raises:
            TypeError: If `hidden_states` is not a `mindspore.Tensor`.
            ValueError: If `hidden_states` is not of the correct dtype.
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


class SeamlessM4TEncoderLayer(nn.Cell):

    """
    The SeamlessM4TEncoderLayer class represents a single layer of the SeamlessM4T model encoder.
    It includes self-attention and feed-forward network components.

    This class inherits from the nn.Cell class and is initialized with a SeamlessM4TConfig object, encoder_ffn_dim,
    and encoder_attention_heads. The class also features a 'construct' method that takes hidden_states and
    attention_mask as input and returns the output tensor.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        self_attn (SeamlessM4TAttention): The self-attention mechanism.
        attn_dropout (nn.Dropout): The dropout layer for attention.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization for self-attention.
        ffn (SeamlessM4TFeedForwardNetwork): The feed-forward network.
        ffn_layer_norm (nn.LayerNorm): The layer normalization for the feed-forward network.
        ffn_dropout (nn.Dropout): The dropout layer for the feed-forward network.

    Methods:
        construct: Applies self-attention and feed-forward operations to the input hidden_states and returns
            the output tensor.

    Args:
        hidden_states (mindspore.Tensor): The input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask (mindspore.Tensor): The attention mask of size (batch, 1, tgt_len, src_len) where padding
            elements are indicated by very large negative values.
        output_attentions (bool, optional): Determines whether to return attention weights. Defaults to False.
    """
    def __init__(self, config: SeamlessM4TConfig, encoder_ffn_dim=None, encoder_attention_heads=None):
        """
        Initializes a new instance of the SeamlessM4TEncoderLayer class.

        Args:
            self: The object itself.
            config (SeamlessM4TConfig): The configuration object for the encoder layer.
            encoder_ffn_dim (int): The dimension of the feed-forward network in the encoder layer. Defaults to None.
            encoder_attention_heads (int): The number of attention heads in the self-attention mechanism of the
                encoder layer. Defaults to None.

        Returns:
            None

        Raises:
            None

        This method initializes the SeamlessM4TEncoderLayer with the given configuration and optional parameters.
        It sets the embed_dim attribute to the hidden size specified in the config object. The self-attention mechanism
        is initialized with the embed_dim, number of attention heads, and dropout rate specified in the config object.
        The attention dropout is set using the dropout rate specified in the config object. The self-attention
        layer normalization is initialized with the embed_dim. The feed-forward network is initialized with the config
        object and the encoder_ffn_dim parameter. The feed-forward network layer normalization is initialized with the
        hidden size specified in the config object. The feed-forward network dropout is set using the activation
        dropout rate specified in the config object.
        """
        super().__init__()
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])

        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)

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


class SeamlessM4TDecoderLayer(nn.Cell):

    """
    SeamlessM4TDecoderLayer represents a decoder layer in the SeamlessM4T model architecture for machine translation.
    This class implements the decoder layer functionality with self-attention, cross-attention, feed-forward network,
    and layer normalization.
    It inherits from nn.Cell and is designed to be used within a larger Transformer model for translation tasks.

    Attributes:
        config (SeamlessM4TConfig): Configuration object containing parameters for the decoder layer.
        decoder_ffn_dim (int, optional): Dimension of the feed-forward network in the decoder layer.
        decoder_attention_heads (int, optional): Number of attention heads in the decoder layer.

    Methods:
        __init__: Initializes the decoder layer with the specified configuration and optional parameters for the
            feed-forward network and attention heads.
        construct: Executes the forward pass of the decoder layer, processing input hidden states and performing
            self-attention, cross-attention with encoder hidden states, and feed-forward network operations.

    Args:
        hidden_states (mindspore.Tensor): Input tensor of shape (batch, seq_len, embed_dim) to the decoder layer.
        attention_mask (mindspore.Tensor, optional): Attention mask tensor of size (batch, 1, tgt_len, src_len) to
            mask padding elements.
        encoder_hidden_states (mindspore.Tensor, optional): Input tensor of shape (batch, seq_len, embed_dim)
            for cross-attention.
        encoder_attention_mask (mindspore.Tensor, optional): Attention mask tensor of size (batch, 1, tgt_len, src_len)
            for encoder attention.
        past_key_value (Tuple[mindspore.Tensor], optional): Cached past key and value projection states for optimization.
        output_attentions (bool, optional): Whether to return attention tensors of all attention layers.

    Returns:
        outputs (Tuple[mindspore.Tensor]): Tuple containing the final hidden states and present key-value states.
          If output_attentions is True, also includes self-attention weights and cross-attention weights.
    """
    def __init__(self, config: SeamlessM4TConfig, decoder_ffn_dim=None, decoder_attention_heads=None):
        """
        Initializes a new instance of the SeamlessM4TDecoderLayer class.

        Args:
            self: The instance of the class.
            config (SeamlessM4TConfig): The configuration object for the decoder layer.
            decoder_ffn_dim (int, optional): The dimension of the feed-forward network in the decoder layer.
                If not provided, the value is taken from the config object.
            decoder_attention_heads (int, optional): The number of attention heads in the decoder layer.
                If not provided, the value is taken from the config object.

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
        self.self_attn = SeamlessM4TAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(p=config.dropout)

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.cross_attention = SeamlessM4TAttention(
            self.embed_dim, decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        self.cross_attention_layer_norm = nn.LayerNorm([self.embed_dim])

        self.ffn = SeamlessM4TFeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

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


############ SUB-MODELS related code ################


class SeamlessM4TPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SeamlessM4TConfig
    base_model_prefix = "seamless_m4t"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SeamlessM4TEncoderLayer", "SeamlessM4TDecoderLayer", "SeamlessM4TConformerEncoderLayer"]

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
        elif isinstance(cell, SeamlessM4TConformerSelfAttention):
            if hasattr(cell, "pos_bias_u"):
                cell.pos_bias_u.set_data(initializer(XavierUniform(),
                                                    cell.pos_bias_u.shape, cell.pos_bias_u.dtype))
            if hasattr(cell, "pos_bias_v"):
                cell.pos_bias_v.set_data(initializer(XavierUniform(),
                                                    cell.pos_bias_v.shape, cell.pos_bias_v.dtype))

        elif isinstance(cell, SeamlessM4TConformerPositionalConvEmbedding):
            cell.conv.weight.set_data(initializer(Normal(2 * math.sqrt(1 / (cell.conv.kernel_size[0] * cell.conv.in_channels))),
                                                    cell.conv.weight.shape, cell.conv.weight.dtype))
            cell.conv.bias.set_data(initializer('zeros', cell.conv.bias.shape, cell.conv.bias.dtype))
        elif isinstance(cell, SeamlessM4TConformerFeatureProjection):
            k = math.sqrt(1 / cell.projection.in_channels)
            cell.projection.weight.set_data(initializer(Uniform(k),
                                        cell.projection.weight.shape, cell.projection.weight.dtype))
            cell.projection.bias.set_data(initializer(Uniform(k),
                                        cell.projection.bias.shape, cell.projection.bias.dtype))

        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Conv1d):
            cell.weight.set_data(initializer(HeNormal(),
                                              cell.weight.shape, cell.weight.dtype))

            if cell.bias is not None:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(Uniform(k),
                                   cell.bias.shape, cell.bias.dtype))

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        """
        Method to compute sub-sample lengths from the attention mask.

        Args:
            self (SeamlessM4TPreTrainedModel): The instance of the class calling the method.
            attention_mask (numpy.ndarray): A 2D numpy array representing the attention mask.

        Returns:
            numpy.ndarray: A 1D array containing the computed sequence lengths after subsampling.

        Raises:
            ValueError: If the provided attention mask is not a valid numpy array.
            TypeError: If the computed sequence lengths cannot be converted to float32 or floored.
        """
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = kernel_size // 2
        seq_lens = attention_mask.shape[1] - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.astype(mindspore.float32).floor()

    def compute_last_hidden_states_per_sample(
        self,
        hidden_states: Tuple[Tuple[mindspore.Tensor]],
        beam_indices: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        """
        Computes the last hidden states.

        Parameters:
            hidden_states (`Tuple[Tuple[mindspore.Tensor]]`):
                The generated hidden states. Tuple (one element for each generated token) of tuples (one element for
                each layer of the decoder) of mindspore.Tensor of shape (batch_size*num_beams*num_return_sequences,
                generated_length, hidden_size).
            beam_indices (`mindspore.Tensor`, *optional*):
                Beam indices of generated token id at each generation step. `mindspore.Tensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.

        Returns:
            `mindspore.Tensor`: A `mindspore.Tensor` of shape
                `(batch_size*num_return_sequences, sequence_length, hidden_size)` containing the last hidden states.
        """
        # 1. First, let's compute last_hidden_states from hidden_states.
        # For each generation step, takes the hidden state from the last layer.
        # shape: (batch_size*vocab_size*num_return_sequences, # generation_steps, hidden_dim)
        last_hidden_states = ops.concat([hidden_states[-1] for hidden_states in hidden_states], axis=1)

        # 2. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        # in that case, return directly last_hidden_states
        if beam_indices is None:
            return last_hidden_states

        # 3. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.copy()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 4. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards anyways
        beam_indices[beam_indices_mask] = 0

        # 5. expand beam_indices to last_hidden_states dim
        beam_indices = beam_indices.unsqueeze(-1)
        beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])

        # 6. select the right candidate for each beam
        # in other words, new_last_hidden_states[i,j,k] = last_hidden_states[beam_indices[i,j,k], j, k] for all i, j, k
        last_hidden_states = ops.gather(last_hidden_states, 0, beam_indices)

        return last_hidden_states


class SeamlessM4TSpeechEncoder(SeamlessM4TPreTrainedModel):

    """
    A class representing a SeamlessM4TSpeechEncoder in Python.

    This class is a part of the SeamlessM4T package and is used for speech encoding tasks. It inherits from the
    SeamlessM4TPreTrainedModel class.

    Attributes:
        feature_projection (SeamlessM4TConformerFeatureProjection): An instance of SeamlessM4TConformerFeatureProjection
            class for feature projection.
        encoder (SeamlessM4TConformerEncoder): An instance of SeamlessM4TConformerEncoder class for encoding.
        intermediate_ffn (SeamlessM4TConformerFeedForward): An instance of SeamlessM4TConformerFeedForward class for
            intermediate feed-forward network.
        adapter (SeamlessM4TConformerAdapter): An optional instance of SeamlessM4TConformerAdapter class for
            adapting hidden states.
        inner_layer_norm (nn.LayerNorm): A layer normalization module.

    Methods:
        __init__: Initializes the SeamlessM4TSpeechEncoder class with the given configuration.
        construct: Constructs the speech encoder.

    Note:
        Make sure to provide either `input_features` or `inputs_embeds` as an argument when calling the
        `construct` method.

    Raises:
        ValueError: If both `input_features` and `inputs_embeds` are `None` in the `construct` method.

    Returns:
        Union[Tuple, Wav2Vec2BaseModelOutput]: The output of the speech encoder, which includes the hidden states,
            encoder hidden states, and attentions.
    """
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4TConfig):
        """
        Initializes a new instance of the SeamlessM4TSpeechEncoder class.

        Args:
            self: The object itself.
            config (SeamlessM4TConfig): The configuration object that contains various settings for the speech encoder.
                This object should be an instance of the SeamlessM4TConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.feature_projection = SeamlessM4TConformerFeatureProjection(config)
        self.encoder = SeamlessM4TConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4TConformerFeedForward(config, act_fn="relu", dropout=0.0)
        self.adapter = SeamlessM4TConformerAdapter(config) if config.add_adapter else None
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
        Constructs the Wav2Vec2 speech encoder.

        Args:
            input_features (Optional[mindspore.Tensor]): The input features to be encoded.
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask for the input features.
                Defaults to None.
            output_attentions (Optional[bool], optional): Whether to return attentions.
                If not provided, it defaults to the value in the model's configuration.
            output_hidden_states (Optional[bool], optional): Whether to return hidden states.
                If not provided, it defaults to the value in the model's configuration.
            return_dict (Optional[bool], optional): Whether to return the output as a dict.
                If not provided, it defaults to the value in the model's configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Tuple, Wav2Vec2BaseModelOutput]: A tuple containing the encoded hidden states and
                optional additional outputs.

        Raises:
            ValueError: If both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.
                Make sure one of them is not `None`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4TSpeechEncoder.forward`.
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
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):

    """
    A class that implements the SeamlessM4TEncoder model, which is used for encoding input sequences in the SeamlessM4T
    framework.

    This class inherits from the SeamlessM4TPreTrainedModel class and provides methods for initializing the encoder and
    performing the encoding process.

    Attributes:
        dropout (float): The dropout probability for the encoder.
        layerdrop (float): The layer dropout probability for the encoder.
        padding_idx (int): The index used for padding tokens.
        embed_dim (int): The dimensionality of the embedding vectors.
        is_t2u_encoder (bool): A flag indicating whether the encoder is used for text_to_units model.
        max_source_positions (int): The maximum number of source positions.
        embed_scale (float): The scale factor for the embedding vectors.
        embed_tokens (nn.Embedding): The embedding layer for the input tokens.
        embed_positions (SeamlessM4TSinusoidalPositionalEmbedding): The positional embedding layer.
        layers (nn.CellList): The list of encoder layers.
        layer_norm (nn.LayerNorm): The layer normalization layer.
        gradient_checkpointing (bool): A flag indicating whether to use gradient checkpointing during training.

    Methods:
        __init__: Initializes the SeamlessM4TEncoder instance.

        construct: Constructs the encoder based on the input arguments and returns the encoded hidden states.
    """
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        """
        Initializes a new instance of the SeamlessM4TEncoder class.

        Args:
            self: The object itself.
            config (SeamlessM4TConfig): An instance of the SeamlessM4TConfig class containing configuration settings.
            embed_tokens (Optional[nn.Embedding]): An optional instance of the nn.Embedding class representing
                embedded tokens.
            is_t2u_encoder (bool): A boolean indicating whether the encoder is used for translation from text to
                utterance.

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

            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4TEncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.CellList(layers)

        self.layer_norm = nn.LayerNorm([config.hidden_size])

        self.gradient_checkpointing = False
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
        elif input_ids is not None:
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
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.forward,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
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


class SeamlessM4TDecoder(SeamlessM4TPreTrainedModel):

    """
    SeamlessM4TDecoder

    This class represents a decoder module for the SeamlessM4T model. It inherits from SeamlessM4TPreTrainedModel and
    implements methods for initializing the decoder, constructing the decoder, and getting/setting input embeddings.

    Attributes:
        config: An instance of SeamlessM4TConfig containing the configuration settings for the decoder.
        dropout: The dropout rate specified in the configuration.
        layerdrop: The layer drop rate specified in the configuration.
        padding_idx: The padding token index specified in the configuration.
        vocab_size: The size of the vocabulary specified in the configuration.
        max_target_positions: The maximum target positions specified in the configuration.
        embed_scale: The scaling factor for embedding specified in the configuration.
        embed_tokens: An instance of nn.Embedding for embedding tokens.
        embed_positions: An instance of SeamlessM4TSinusoidalPositionalEmbedding for embedding positions.
        layers: A list of SeamlessM4TDecoderLayer instances representing the decoder layers.
        layer_norm: An instance of nn.LayerNorm for layer normalization.
        gradient_checkpointing: A boolean specifying whether gradient checkpointing is enabled.

    Methods:
        __init__: Initializes the SeamlessM4TDecoder with the given configuration and embed_tokens.
        get_input_embeddings: Returns the input embeddings.
        set_input_embeddings: Sets the input embeddings for the decoder.
        construct: Constructs the decoder with the given input and optional arguments.

    Args:
        input_ids: A mindspore.Tensor of shape (batch_size, sequence_length) representing input sequence token indices.
        attention_mask: A mindspore.Tensor of shape (batch_size, sequence_length) representing attention mask to avoid
            padding tokens.
        encoder_hidden_states: A mindspore.Tensor of shape (batch_size, encoder_sequence_length, hidden_size)
            representing hidden states of the encoder.
        encoder_attention_mask: A mindspore.Tensor of shape (batch_size, encoder_sequence_length) representing
            attention mask for cross-attention.
        past_key_values: A tuple of tuples of mindspore.Tensor representing pre-computed hidden-states for sequential
            decoding.
        inputs_embeds: A mindspore.Tensor of shape (batch_size, sequence_length, hidden_size) representing embedded
            input representation.
        use_cache: A boolean specifying whether to use cache for sequential decoding.
        output_attentions: A boolean specifying whether to return attentions tensors of all attention layers.
        output_hidden_states: A boolean specifying whether to return hidden states of all layers.
        return_dict: A boolean specifying whether to return a ModelOutput instead of a plain tuple.
    """
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        """
        Initializes an instance of the 'SeamlessM4TDecoder' class.

        Args:
            self: An instance of the 'SeamlessM4TDecoder' class.
            config (SeamlessM4TConfig): An object containing configuration options for the decoder.
            embed_tokens (Optional[nn.Embedding]): An optional embedding object to be used for token embeddings.

        Returns:
            None

        Raises:
            None

        This method initializes the 'SeamlessM4TDecoder' instance by setting various attributes and creating necessary
        objects. It takes the following parameters:

        - self: An instance of the 'SeamlessM4TDecoder' class.
        - config (SeamlessM4TConfig): An object that holds configuration options for the decoder.
        It provides access to various hyperparameters and settings.
        - embed_tokens (Optional[nn.Embedding]): An optional embedding object that can be used for token embeddings.
        If provided, the 'embed_tokens' attribute of the decoder will be set to this object. Otherwise, a new embedding
        object will be created using the 'vocab_size' and 'hidden_size' from the 'config' object.

        Note:
            The 'config' parameter is mandatory, while the 'embed_tokens' parameter is optional.

        The method performs the following actions:

        1. Calls the superclass '__init__' method with the 'config' parameter.
        2. Sets the 'dropout' attribute to the 'dropout' value from the 'config' object.
        3. Sets the 'layerdrop' attribute to the 'decoder_layerdrop' value from the 'config' object.
        4. Sets the 'padding_idx' attribute to the 'pad_token_id' value from the 'config' object.
        5. Sets the 'vocab_size' attribute to the 'vocab_size' value from the 'config' object.
        6. Sets the 'max_target_positions' attribute to the 'max_position_embeddings' value from the 'config' object.
        7. Sets the 'embed_scale' attribute based on the 'scale_embedding' value from the 'config' object.
        If 'scale_embedding' is True, it sets 'embed_scale' to the square root of 'hidden_size'; otherwise, it
        sets 'embed_scale' to 1.0.
        8. If 'embed_tokens' is not None:

            - Creates a new 'nn.Embedding' object named 'self.embed_tokens' with 'embed_tokens.vocab_size',
            'embed_tokens.embedding_size', and 'self.padding_idx' as arguments.
            - Sets the weight of 'self.embed_tokens' to the weight of 'embed_tokens'.
        9. If 'embed_tokens' is None:

            - Creates a new 'nn.Embedding' object named 'self.embed_tokens' with 'self.vocab_size', 'config.hidden_size',
            and 'self.padding_idx' as arguments.
        10. Creates a 'SeamlessM4TSinusoidalPositionalEmbedding' object named 'self.embed_positions' with
        'self.max_target_positions', 'config.hidden_size', and 'self.padding_idx' as arguments.
        11. Creates a list named 'layers'.
        12. Iterates 'config.decoder_layers' times and appends a 'SeamlessM4TDecoderLayer' object to 'layers',
        using 'config', 'config.decoder_attention_heads', and 'config.decoder_ffn_dim' as arguments.
        13. Sets the 'layers' attribute to a 'nn.CellList' containing the objects in 'layers'.
        14. Creates a 'nn.LayerNorm' object named 'self.layer_norm' with a list containing 'config.hidden_size'
        as the argument.
        15. Sets the 'gradient_checkpointing' attribute to False.
        16. Calls the 'post_init' method.

        Note: The 'post_init' method is not defined in the given code snippet.
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

        self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for _ in range(config.decoder_layers):
            layers.append(
                SeamlessM4TDecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                )
            )
        self.layers = nn.CellList(layers)
        self.layer_norm = nn.LayerNorm([config.hidden_size])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the SeamlessM4TDecoder.

        Args:
            self (SeamlessM4TDecoder): An instance of the SeamlessM4TDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TDecoder.

        Args:
            self (SeamlessM4TDecoder): The instance of SeamlessM4TDecoder.
            value: The input embeddings to be set.

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
        elif input_ids is not None:
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

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


class SeamlessM4TTextToUnitModel(SeamlessM4TPreTrainedModel):

    """
    This class represents a text-to-unit (T2U) model for seamless conversion and inference between natural language text
    and MindSpore tensor units. It inherits functionality from the SeamlessM4TPreTrainedModel class and provides methods
    for initializing the model and constructing the T2U conversion process using encoder and decoder components.
    The class includes configurable parameters for input, attention, and output settings, as well as the option to
    return a dictionary of model outputs. The model supports the use of cached values and the generation of hidden
    states and attentions.
    """
    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        """
        Initializes an instance of the SeamlessM4TTextToUnitModel class.

        Args:
            self: The instance of the class.
            config (SeamlessM4TConfig): The configuration object for the model.
            embed_tokens_decoder (Optional[nn.Embedding]): An optional embedding layer for the decoder.
                Default value is None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.encoder = SeamlessM4TEncoder(config, is_t2u_encoder=True)
        self.decoder = SeamlessM4TDecoder(config, embed_tokens_decoder)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        """
        This method 'construct' in the class 'SeamlessM4TTextToUnitModel' constructs the text-to-unit model and takes
        the following parameters:

        Args:
            self: Represents the instance of the class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of input sequence tokens
                in the vocabulary.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor indicating which tokens should be
                attended to and which should not.
            decoder_input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of decoder
                input sequence tokens in the vocabulary.
            decoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the decoder.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]]): The output from the encoder model.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values for the decoder.
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings for the encoder.
            decoder_inputs_embeds (Optional[mindspore.Tensor]): The input embeddings for the decoder.
            use_cache (Optional[bool]): Flag indicating whether to use caching.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            return_dict (Optional[bool]): Flag indicating whether to use return dict.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]: The return value can be a tuple of tensors or
                an instance of Seq2SeqModelOutput, representing the output of the text-to-unit model.

        Raises:
            None
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
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


class SeamlessM4TTextToUnitForConditionalGeneration(SeamlessM4TPreTrainedModel):

    """
    This class represents a SeamlessM4TTextToUnitForConditionalGeneration model for conditional text generation.
    It is a subclass of SeamlessM4TPreTrainedModel.

    The class provides methods for initializing the model, getting the encoder and decoder, setting the output and
    input embeddings, constructing the model, preparing inputs for generation, preparing decoder input ids from labels,
    reordering cache, and tying weights.

    Attributes:
        config (SeamlessM4TConfig): The configuration object for the model.
        model (SeamlessM4TTextToUnitModel): The SeamlessM4TTextToUnitModel instance used for text-to-unit conversion.
        lm_head (nn.Dense): The linear layer for generating the language model output.

    Methods:
        __init__:
            Initializes the SeamlessM4TTextToUnitForConditionalGeneration model.
        get_encoder:
            Returns the encoder of the model.
        get_decoder:
            Returns the decoder of the model.
        get_output_embeddings:
            Returns the output embeddings of the model.
        set_output_embeddings:
            Sets the output embeddings of the model.
        get_input_embeddings:
            Returns the input embeddings of the model.
        set_input_embeddings:
            Sets the input embeddings of the model.
        construct:
            Constructs the model for conditional text generation.
        prepare_inputs_for_generation:
            Prepares the inputs for text generation.
        prepare_decoder_input_ids_from_labels:
            Prepares the decoder input ids from labels.
        _reorder_cache:
            Reorders the cache based on the beam index.
        _tie_weights:
            Ties the input and output embeddings weights if the configuration allows.

    Note:
        The class inherits from SeamlessM4TPreTrainedModel and extends its functionality for conditional text generation.
    """
    _keys_to_ignore_on_load_missing = [
        "vocoder",
        "speech_encoder",
        "text_encoder",
        "text_decoder",
    ]
    _tied_weights_keys = ["decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(
        self,
        config: SeamlessM4TConfig,
        embed_tokens_decoder: Optional[nn.Embedding] = None,
    ):
        """
        Initializes a new instance of the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self: The object itself.
            config (SeamlessM4TConfig): The configuration for the model.
            embed_tokens_decoder (Optional[nn.Embedding]): The decoder for embedding tokens. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        # update config - used principaly for bos_token_id etc.
        config = copy.deepcopy(config)
        for param, val in config.to_dict().items():
            if param.startswith("t2u_"):
                config.__setattr__(param[4:], val)
        super().__init__(config)

        self.model = SeamlessM4TTextToUnitModel(config, embed_tokens_decoder)

        self.lm_head = nn.Dense(config.hidden_size, config.t2u_vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Method to retrieve the encoder from the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self: This parameter refers to the instance of the class. It is required for accessing the attributes and
                methods of the class.

        Returns:
            encoder: This method returns the encoder from the model associated with the class.
                The encoder is a component that encodes input data into a different representation.

        Raises:
            None.
        """
        return self.model.encoder

    def get_decoder(self):
        """
        This method returns the decoder model for the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self: A reference to the current instance of the class.

        Returns:
            decoder: This method returns the decoder model for the SeamlessM4TTextToUnitForConditionalGeneration class.

        Raises:
            None.
        """
        return self.model.decoder

    def get_output_embeddings(self):
        """
        get_output_embeddings method in the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self: The instance of the class.

        Returns:
            lm_head: This method returns the lm_head attribute of the class, which represents the output embeddings.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self (SeamlessM4TTextToUnitForConditionalGeneration): The instance of the class.
            new_embeddings: The new embeddings to be set for the output.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the SeamlessM4TTextToUnitForConditionalGeneration model
        for conditional generation.

        Args:
            self (SeamlessM4TTextToUnitForConditionalGeneration): The instance of the
                SeamlessM4TTextToUnitForConditionalGeneration class.

        Returns:
            embed_tokens: This method returns the input embeddings for the model's decoder.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self (SeamlessM4TTextToUnitForConditionalGeneration): The instance of the class.
            value: The input embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

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
        """
        Constructs a text-to-unit model for conditional generation in SeamlessM4TTextToUnitForConditionalGeneration class.

        Args:
            self: The object itself.
            input_ids (mindspore.Tensor, optional): The input tensor of shape [batch_size, sequence_length]
                representing the input sequence. Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor of shape
                [batch_size, sequence_length] representing the attention mask. Default is None.
            decoder_input_ids (mindspore.Tensor, optional): The decoder input tensor of shape
                [batch_size, sequence_length] representing the decoder input sequence. Default is None.
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask tensor of shape
                [batch_size, sequence_length] representing the decoder attention mask. Default is None.
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs tensor. Default is None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values tensor. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The embedded input tensor of shape
                [batch_size, sequence_length, hidden_size] representing the embedded inputs. Default is None.
            decoder_inputs_embeds (mindspore.Tensor, optional): The embedded decoder input tensor of shape
                [batch_size, sequence_length, hidden_size] representing the embedded decoder inputs. Default is None.
            labels (mindspore.Tensor, optional): The labels tensor of shape [batch_size, sequence_length]
                representing the labels for training. Default is None.
            use_cache (bool, optional): Whether to use cache. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return dictionary. Default is None.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
                The output of the model.

                - If return_dict is False, it returns a tuple containing the masked language model loss
                (if labels is not None) and the model outputs.
                - If return_dict is True, it returns a Seq2SeqLMOutput object containing the masked language
                model loss, logits, past key values, decoder hidden states, decoder attentions, cross attentions,
                encoder last hidden state, encoder hidden states, and encoder attentions.

        Raises:
            None.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
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

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
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
            self: The instance of the class.
            decoder_input_ids (Tensor): The input ids for the decoder. It is the input sequence tensor of token indices.
                Shape: (batch_size, sequence_length)
            past_key_values (Tuple, optional): The previously calculated key and value tensors for fast decoding.
                Default: None.
            attention_mask (Tensor, optional): The attention mask tensor.
                It is a binary tensor indicating the position of the padded tokens.
                Value 1 indicates a valid token, and value 0 indicates a padded token.
                Shape: (batch_size, sequence_length)
            use_cache (bool, optional): Whether to use the cache for fast decoding.
                Default: None.
            encoder_outputs (ModelOutput, optional): The outputs of the encoder model.
                Default: None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (None): Always set to None.
                - 'encoder_outputs' (ModelOutput): The outputs of the encoder model.
                - 'past_key_values' (Tuple, optional): The previously calculated key and value tensors for fast decoding.
                - 'decoder_input_ids' (Tensor): The input ids for the decoder after processing.
                - 'attention_mask' (Tensor, optional): The attention mask tensor.
                - 'use_cache' (bool, optional): Whether to use the cache for fast decoding.

        Raises:
            ValueError: If the shape of decoder_input_ids is invalid.
            TypeError: If encoder_outputs is not of type ModelOutput.
            TypeError: If past_key_values is not of type Tuple.
            TypeError: If attention_mask is not of type Tensor.
            TypeError: If use_cache is not of type bool.
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

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare the decoder input ids from labels.

        Args:
            self (SeamlessM4TTextToUnitForConditionalGeneration): An instance of the
                SeamlessM4TTextToUnitForConditionalGeneration class.
            labels (mindspore.Tensor): The labels for the decoder input. A tensor containing the token ids.

        Returns:
            None: This method modifies the decoder input ids in-place.

        Raises:
            None.
        """
        return shift_tokens_right(labels, self.config.t2u_pad_token_id, self.config.t2u_decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache of past key values based on the specified beam indices.

        Args:
            past_key_values (tuple): A tuple containing the cache of past key values. Each element of the tuple
                represents the past key values for a specific layer. Each layer's past key values is further
                represented as a tuple containing three elements:

                1. A tensor representing the past states for the current layer.
                2. A tensor representing the past attentions for the current layer.
                3. A tensor representing the past cross-attentions for the current layer.
            beam_idx (tensor): A tensor containing the indices of the selected beams.

        Returns:
            tuple: A tuple containing the reordered cache of past key values. Each element of the tuple represents
                the reordered past key values for a specific layer.
                Each layer's reordered past key values is further represented as a tuple containing three elements:

                1. A tensor representing the reordered past states for the current layer.
                2. A tensor representing the reordered past attentions for the current layer.
                3. A tensor representing the reordered past cross-attentions for the current layer.

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

    def _tie_weights(self) -> None:
        """
        Tie the weights of the input and output embeddings in the 'SeamlessM4TTextToUnitForConditionalGeneration' model.

        Args:
            self: An instance of the 'SeamlessM4TTextToUnitForConditionalGeneration' class.

        Returns:
            None.

        Raises:
            None.

        This method checks if the 'tie_word_embeddings' attribute is present in the 'config' object of the model.
        If it is present and set to True (default), it ties the weights of the output embeddings with the input
        embeddings. The 'tie_or_clone_weights' function is used to perform the weight tying operation.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())


############ VOCODER related code ################


HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [ops.nn.Cell](https://pyops.org/docs/stable/nn.html#ops.nn.Cell) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SeamlessM4TConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Cell):

    """
    This class represents a High Fidelity Generative Adversarial Network (HifiGan) Residual Block.
    It is a subclass of nn.Cell and is used in the construction of the HifiGan model.

    Attributes:
        channels (int): The number of input and output channels for the convolutional layers.
        kernel_size (int): The size of the convolutional kernel.
        dilation (tuple): The dilation factors to be applied to the convolutional layers.
        leaky_relu_slope (float): The slope of the negative region of the leaky ReLU activation function.

    Methods:
        __init__:
            Initializes a new instance of the HifiGanResidualBlock class.

        get_padding:
            Calculates the padding to be applied to the convolutional layers.

        apply_weight_norm:
            Applies weight normalization to the convolutional layers.

        remove_weight_norm:
            Removes weight normalization from the convolutional layers.

        construct:
            Constructs the HifiGanResidualBlock by applying the convolutional layers and residual connections to
            the input hidden states.

    Note:
        The HifiGanResidualBlock class inherits from nn.Cell, which is a base class for all neural network modules
        in MindSpore. It provides basic functionalities for constructing and managing neural networks.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        """
        __init__

        Initializes a new instance of the HifiGanResidualBlock class.

        Args:
            channels (int): The number of input and output channels for the convolutional layers.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            dilation (tuple of int, optional): The dilation rates for the convolutional layers. Defaults to (1, 3, 5).
            leaky_relu_slope (float, optional): The slope for the Leaky ReLU activation function. Defaults to 0.1.

        Returns:
            None.

        Raises:
            ValueError: If channels, kernel_size, or any element in the dilation tuple is less than or equal to 0.
            TypeError: If the provided values for channels, kernel_size, dilation, or leaky_relu_slope are not of
                the expected types.
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
        Returns the required padding size for a given kernel size and dilation factor.

        Args:
            self (HifiGanResidualBlock): An instance of the HifiGanResidualBlock class.
            kernel_size (int): The size of the kernel.
            dilation (int, optional): The dilation factor (default is 1).

        Returns:
            int: The calculated padding size.

        Raises:
            None.

        This method calculates the required padding size based on the given kernel size and dilation factor.
        The padding size is determined by the formula: (kernel_size * dilation - dilation) // 2. The method
        then returns the calculated padding size as an integer value.
        """
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        """
        Apply weight normalization to the convolutional layers in the HifiGanResidualBlock.

        Args:
            self: The instance of the HifiGanResidualBlock class.

        Returns:
            None.

        Raises:
            None.
        """
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layers within the HifiGanResidualBlock.

        Args:
            self: An instance of the HifiGanResidualBlock class.

        Returns:
            None.

        Raises:
            None.
        """
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def construct(self, hidden_states):
        """
        Constructs a single residual block in the HifiGan model.

        Args:
            self (HifiGanResidualBlock): The instance of the HifiGanResidualBlock class.
            hidden_states (Tensor): The input hidden states for the residual block.
                Expected shape is [batch_size, channels, sequence_length].

        Returns:
            None

        Raises:
            None
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class SeamlessM4TVariancePredictor(nn.Cell):

    """
    This class represents a variance predictor module used in the SeamlessM4T model. It is a subclass of the nn.Cell class.

    Attributes:
        conv1 (nn.Conv1d): A 1-dimensional convolutional layer that processes the input hidden states.
        activation_function (nn.ReLU): The activation function applied after the first convolutional layer.
        ln1 (nn.LayerNorm): Layer normalization applied to the output of the first convolutional layer.
        dropout_module (nn.Dropout): Dropout module that applies dropout to the normalized hidden states.
        conv2 (nn.Conv1d): A second 1-dimensional convolutional layer that further processes the hidden states.
        ln2 (nn.LayerNorm): Layer normalization applied to the output of the second convolutional layer.
        proj (nn.Dense): A fully connected layer that projects the hidden states to a single output dimension.

    Methods:
        construct:
            Applies the variance predictor module to the input hidden states.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the SeamlessM4TVariancePredictor class.

        Args:
            self: The object instance.
            config:
                An object of type 'Config' that contains the configuration parameters for the variance predictor.

                - unit_embed_dim (int): The dimension of the input embeddings.
                - variance_predictor_kernel_size (int): The size of the kernel for the convolutional layers.
                - var_pred_dropout (float): The dropout probability for the dropout layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        embed_dim = config.unit_embed_dim
        kernel_size = config.variance_predictor_kernel_size
        var_pred_dropout = config.var_pred_dropout

        self.conv1 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            pad_mode='pad',
            padding=(kernel_size - 1) // 2,
        )
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm([embed_dim])
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            pad_mode='pad',
            padding=1,
        )
        self.ln2 = nn.LayerNorm([embed_dim])
        self.proj = nn.Dense(embed_dim, 1)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the SeamlessM4TVariancePredictor by processing the hidden_states tensor.

        Args:
            self (SeamlessM4TVariancePredictor): An instance of the SeamlessM4TVariancePredictor class.
            hidden_states (mindspore.Tensor): A tensor representing the hidden states.

        Returns:
            mindspore.Tensor: A tensor representing the processed hidden states.

        Raises:
            ValueError: If the hidden_states tensor is invalid or has incompatible dimensions.
            RuntimeError: If an error occurs during the processing of the hidden_states tensor.
        """
        # Input: B x T x C; Output: B x T
        hidden_states = self.conv1(hidden_states.swapaxes(1, 2))
        hidden_states = self.activation_fuction(hidden_states).swapaxes(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        hidden_states = self.conv2(hidden_states.swapaxes(1, 2))
        hidden_states = self.activation_fuction(hidden_states).swapaxes(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(axis=2)


class SeamlessM4THifiGan(nn.Cell):

    """
    This class represents a SeamlessM4THifiGan, a neural network model for converting log-mel spectrograms into
    speech waveforms.

    The class inherits from nn.Cell and contains methods for initializing the model and constructing the speech
    waveform from input log-mel spectrograms.

    Attributes:
        `config`: The configuration object containing various model parameters such as embedding dimensions,
            kernel sizes, and stride rates.
        `leaky_relu_slope`: The slope value for the leaky ReLU activation function.
        `num_kernels`: The number of kernels in the model's resblocks.
        `num_upsamples`: The number of upsampling layers in the model.
        `conv_pre`: The pre-convolution layer that takes in the input log-mel spectrograms.
        `upsampler`: A list of upsampling layers.
        `resblocks`: A list of HifiGanResidualBlock layers used in the model.
        `conv_post`: The post-convolution layer that transforms the hidden states into the speech waveform.

    Methods:
        `__init__`: Initializes the SeamlessM4THifiGan model with the given configuration.
        `construct`: Converts log-mel spectrograms into speech waveforms.

    Usage:
        To use the SeamlessM4THifiGan model, create an instance of the class with a `config` object, then call the
        `construct` method passing in the input log-mel spectrograms.

    Example:
        ```python
        >>> config = SeamlessM4TConfig(...)
        >>> model = SeamlessM4THifiGan(config)
        >>> waveform = model.construct(input_embeds)
        ```

    Note:
        - The input log-mel spectrograms can be batched or un-batched, and the resulting speech waveform will have
        the corresponding shape.
        - The `construct` method returns a mindspore.Tensor object containing the speech waveform.

    """
    def __init__(self, config: SeamlessM4TConfig):
        """
        Initializes a new instance of the SeamlessM4THifiGan class.

        Args:
            self: The object itself.
            config (SeamlessM4TConfig):
                The configuration object containing various parameters for the model initialization.

                - unit_embed_dim (int): The dimension of the unit embedding.
                - lang_embed_dim (int): The dimension of the language embedding.
                - spkr_embed_dim (int): The dimension of the speaker embedding.
                - leaky_relu_slope (float): The slope of the leaky ReLU activation function.
                - resblock_kernel_sizes (list[int]): The list of kernel sizes for the residual blocks.
                - upsample_rates (list[int]): The list of upsample rates for the transposed convolutions.
                - upsample_kernel_sizes (list[int]): The list of kernel sizes for the transposed convolutions.
                - upsample_initial_channel (int): The initial number of channels for the upsample convolutions.
                - resblock_dilation_sizes (list[int]): The list of dilation sizes for the residual blocks.

        Returns:
            None

        Raises:
            None
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


class SeamlessM4TCodeHifiGan(PreTrainedModel):

    """
    This class represents a high fidelity generative adversarial network (HiFi-GAN) model for seamless text-to-speech
    synthesis in the SeamlessM4T framework. The model includes components for duration prediction, unit embeddings,
    speaker embeddings, language embeddings, and the HiFi-GAN architecture.

    The class includes methods for computing output lengths after the duration layer and the HiFi-GAN convolutional
    layers. It also provides functionality for constructing the model using input sequences, speaker IDs, and language
    IDs, and initializing and applying weight normalization to the model's components.

    The class inherits from PreTrainedModel and contains methods for weight initialization, applying weight
    normalization, and removing weight normalization from the HiFi-GAN components. Additionally, it includes utility
    functions for weight normalization operations.

    For detailed information on each method and its parameters, please refer to the method docstrings within the
    class definition.
    """
    config_class = SeamlessM4TConfig
    main_input_name = "input_embeds"
    _no_split_modules = []

    def __init__(self, config):
        """
        Initializes the SeamlessM4TCodeHifiGan class.

        Args:
            self: The instance of the class.
            config: A configuration object that contains various settings and parameters for the HifiGan model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.pad_token_id = config.t2u_pad_token_id
        self.dur_predictor = SeamlessM4TVariancePredictor(config)

        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)

        self.hifi_gan = SeamlessM4THifiGan(config)

        # Initialize weights and apply final processing
        self.post_init()

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

    def _get_output_hifigan_lengths(self, input_lengths: Union[mindspore.Tensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            # 1D convolutional layer output length formula taken
            # from https://pyops.org/docs/stable/generated/ops.nn.Conv1d.html
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

    def construct(
        self, input_ids: mindspore.Tensor, spkr_id: mindspore.Tensor, lang_id: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            spkr_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        hidden_states = self.unit_embedding(input_ids).swapaxes(1, 2)
        spkr = self.speaker_embedding(spkr_id).swapaxes(1, 2)
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

            # hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).swapaxes(1, 2)
            hidden_states = ops.stack(hidden_states).swapaxes(1, 2)

        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = ops.cat([lang, hidden_states, spkr], axis=1)

        hidden_states = self.hifi_gan(hidden_states)

        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)

        return hidden_states, lengths

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

    def apply_weight_norm(self):
        """
        Applies weight normalization to the layers of the SeamlessM4TCodeHifiGan model.

        Args:
            self: An instance of the SeamlessM4TCodeHifiGan class.

        Returns:
            None: This method modifies the model's layers in-place.

        Raises:
            None.

        This method applies weight normalization to the layers of the HifiGan model within the SeamlessM4TCodeHifiGan
        class. It iterates through each layer and applies weight normalization using the nn.utils.weight_norm() function.

        The layers that are subjected to weight normalization are:

        - self.hifi_gan.conv_pre: Convolutional layer before upsampling.
        - self.hifi_gan.upsampler: List of upsampling layers.
        - self.hifi_gan.resblocks: List of residual blocks.
        - self.hifi_gan.conv_post: Convolutional layer after upsampling.

        The weight normalization technique normalizes the weights of each layer, making the training process more
        stable and accelerating the convergence. It helps to reduce the internal covariate shift and improves the
        generalization performance of the model.

        Note:
            The method modifies the original model's layers and does not return any value.
        """
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    def remove_weight_norm(self):
        """
        Removes weight normalization from the specified layers in the SeamlessM4TCodeHifiGan class.

        Args:
            self: An instance of the SeamlessM4TCodeHifiGan class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method removes weight normalization from the layers in the HifiGan model.
            The following layers are affected:

            - self.hifi_gan.conv_pre: This is the convolutional layer before the upsampling layers.
            - self.hifi_gan.upsampler: These are the upsampling layers in the HifiGan model.
            - self.hifi_gan.resblocks: These are the residual blocks in the HifiGan model.
            - self.hifi_gan.conv_post: This is the convolutional layer after the upsampling layers.

        Note:
            Weight normalization is a technique used in deep learning to normalize the weights of a neural network layer.
            Removing weight normalization can improve the performance or stability of the model in certain scenarios.
        """
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)


############ WHOLE MODEL related code ################
class SeamlessM4TForTextToText(SeamlessM4TPreTrainedModel):

    """
    This class represents a trained model for text-to-text tasks using the SeamlessM4T architecture.
    It is designed for translating text from one language to another.

    The `SeamlessM4TForTextToText` class inherits from the `SeamlessM4TPreTrainedModel` class,
    which provides the basic functionality for a pre-trained model.

    The class has the following attributes:

    - `shared`: An embedding layer that is shared between the encoder and decoder.
    - `text_encoder`: An instance of the `SeamlessM4TEncoder` class, which encodes the input text.
    - `text_decoder`: An instance of the `SeamlessM4TDecoder` class, which decodes the input text.
    - `lm_head`: A linear layer that maps the hidden state to the vocabulary size.
    - Other inherited attributes from `SeamlessM4TPreTrainedModel`.

    The class provides the following methods:

    - `get_encoder()`: Returns the text encoder.
    - `get_decoder()`: Returns the text decoder.
    - `get_output_embeddings()`: Returns the output embeddings.
    - `set_output_embeddings(new_embeddings)`: Sets the output embeddings to the given `new_embeddings`.
    - `get_input_embeddings()`: Returns the input embeddings.
    - `set_input_embeddings(value)`: Sets the input embeddings to the given `value`.
    - `_tie_weights()`: Ties the weights of the word embeddings if specified in the configuration.
    - `construct()`: Constructs the model by encoding the input text and decoding it to generate output.
    - `generate()`: Generates sequences of token ids based on the input text.
    - `prepare_inputs_for_generation()`: Prepares the inputs for generation.

    For more details on the parameters and return values of each method, please refer to the method docstrings.

    Note:
        It is important to specify the target language (`tgt_lang`) or provide correct `text_decoder_input_ids`
        for correct generation.
    """
    _keys_to_ignore_on_load_missing = ["speech_encoder", "t2u_model", "vocoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        """
        Initializes an instance of the SeamlessM4TForTextToText class.

        Args:
            self: The instance of the class itself.
            config (SeamlessM4TConfig): An object of the SeamlessM4TConfig class containing the configuration parameters.
                This parameter is used to define the model's behavior and settings.
                It is required to properly initialize the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Get the text encoder used in the SeamlessM4TForTextToText class.

        Args:
            self: An instance of the SeamlessM4TForTextToText class.

        Returns:
            text_encoder: This method returns the text_encoder attribute of the SeamlessM4TForTextToText instance.

        Raises:
            None.
        """
        return self.text_encoder

    def get_decoder(self):
        """
        Returns the text decoder used by the SeamlessM4TForTextToText class.

        Args:
            self: An instance of the SeamlessM4TForTextToText class.

        Returns:
            text_decoder: This method returns the text decoder associated with the SeamlessM4TForTextToText instance.

        Raises:
            None.

        """
        return self.text_decoder

    def get_output_embeddings(self):
        """
        This method returns the output embeddings from the language model head.

        Args:
            self: An instance of the SeamlessM4TForTextToText class.

        Returns:
            lm_head: This method returns the output embeddings from the language model head.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the SeamlessM4TForTextToText model.

        Args:
            self (SeamlessM4TForTextToText): An instance of the SeamlessM4TForTextToText model.
            new_embeddings (torch.nn.Module): The new output embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.

        This method sets the output embeddings of the model to the given new_embeddings.
        The new_embeddings should be an instance of the torch.nn.Module class. This method does not return any value.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the SeamlessM4TForTextToText model.

        Args:
            self: An instance of the SeamlessM4TForTextToText class.

        Returns:
            embed_tokens: The method returns the input embeddings for the SeamlessM4TForTextToText model.

        Raises:
            None.
        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TForTextToText model.

        Args:
            self (SeamlessM4TForTextToText): The instance of the SeamlessM4TForTextToText class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a torch.Tensor representing the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        """
        Ties the weights of the word embeddings and language modeling head in the 'SeamlessM4TForTextToText' model.

        Args:
            self: An instance of the 'SeamlessM4TForTextToText' class.

        Returns:
            None.

        Raises:
            None.
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
        Constructs the SeamlessM4TForTextToText model.

        Args:
            self (SeamlessM4TForTextToText): The instance of the SeamlessM4TForTextToText class.
            input_ids (mindspore.Tensor, optional): The input tensor of shape [batch_size, sequence_length].
            attention_mask (mindspore.Tensor, optional): The attention mask tensor of shape
                [batch_size, sequence_length].
            decoder_input_ids (mindspore.Tensor, optional): The decoder input tensor of shape
                [batch_size, sequence_length].
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask tensor of shape
                [batch_size, sequence_length].
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs tensor.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values tensor.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings tensor of shape
                [batch_size, sequence_length, embedding_size].
            decoder_inputs_embeds (mindspore.Tensor, optional): The decoder input embeddings tensor of shape
                [batch_size, sequence_length, embedding_size].
            labels (mindspore.Tensor, optional): The labels tensor of shape [batch_size, sequence_length].
            use_cache (bool, optional): Whether to use cache. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a Seq2SeqLMOutput object. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: A Union type object that represents either a
                Seq2SeqLMOutput or a tuple of mindspore.Tensor.

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
        Prepare inputs for text generation.

        Args:
            self (SeamlessM4TForTextToText): An instance of the SeamlessM4TForTextToText class.
            decoder_input_ids (tensor): The input tensor for the decoder model.
            past_key_values (tuple, optional): A tuple containing the past key values for generation.
            attention_mask (tensor, optional): The attention mask tensor.
            use_cache (bool, optional): Whether to use cache for generation.
            encoder_outputs (tensor, optional): The output tensor from the encoder model.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (None): The input tensor IDs.
                - 'encoder_outputs' (tensor): The output tensor from the encoder model.
                - 'past_key_values' (tuple): The past key values for generation.
                - 'decoder_input_ids' (tensor): The input tensor for the decoder model.
                - 'attention_mask' (tensor): The attention mask tensor.
                - 'use_cache' (bool): Whether to use cache for generation.

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
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each element of the tuple represents the past key values for a specific layer.
                The past key values for each layer consist of a tuple of two tensors and one tensor.
                The first tensor represents the past states for tokens, the second tensor represents the past states
                for attentions, and the third tensor represents the past states for cross attentions.
            beam_idx (torch.Tensor): A tensor representing the beam index.

        Returns:
            tuple: A tuple containing the reordered past key values.
                Each element of the tuple represents the reordered past key values for a specific layer.
                The reordered past key values for each layer consist of a tuple of two tensors and one tensor.
                The first tensor represents the reordered past states for tokens, the second tensor represents the
                reordered past states for attentions, and the third tensor represents the reordered past states for
                cross attentions.

        Raises:
            None

        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class SeamlessM4TForSpeechToText(SeamlessM4TPreTrainedModel):

    """
    This class represents a SeamlessM4T model for speech-to-text translation. It is a subclass of
    SeamlessM4TPreTrainedModel.

    The class includes the following methods:

    - `__init__(self, config: SeamlessM4TConfig)`: Initializes the model with the given configuration.
    - `get_encoder(self)`: Returns the speech encoder of the model.
    - `get_decoder(self)`: Returns the text decoder of the model.
    - `get_output_embeddings(self)`: Returns the output embeddings of the model.
    - `set_output_embeddings(self, new_embeddings)`:
    Sets the output embeddings of the model with the given new embeddings.
    - `get_input_embeddings(self)`: Returns the input embeddings of the model.
    - `set_input_embeddings(self, value)`: Sets the input embeddings of the model with the given value.
    - `_tie_weights(self)`: Ties the word embeddings if specified in the configuration.
    - `construct(self, input_features, attention_mask, decoder_input_ids, decoder_attention_mask, encoder_outputs,
    past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states,
    return_dict, **kwargs)`: Constructs the model with the given inputs and returns the output.
    - `generate(self, input_features, tgt_lang, generation_config, logits_processor, stopping_criteria,
    prefix_allowed_tokens_fn, synced_gpus, **kwargs)`: Generates sequences of token ids based on the given input
    features and target language.
    - `prepare_inputs_for_generation(self, decoder_input_ids, past_key_values, attention_mask, use_cache,
    encoder_outputs, **kwargs)`: Prepares the inputs for generation.

    Please refer to the method docstrings for more detailed information on each method's parameters and return values.
    """
    _keys_to_ignore_on_load_missing = ["text_decoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        """
        Initializes an instance of the SeamlessM4TForSpeechToText class.

        Args:
            self (SeamlessM4TForSpeechToText): The instance of the SeamlessM4TForSpeechToText class.
            config (SeamlessM4TConfig): An instance of the SeamlessM4TConfig class containing configuration settings.
                This parameter is used to configure the model with specific settings.
                It is expected to be an object of type SeamlessM4TConfig.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Returns the speech encoder of the SeamlessM4TForSpeechToText class.

        Args:
            self: An instance of the SeamlessM4TForSpeechToText class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.speech_encoder

    def get_decoder(self):
        """
        This method returns the text decoder for the SeamlessM4TForSpeechToText class.

        Args:
            self: The instance of the SeamlessM4TForSpeechToText class.

        Returns:
            text_decoder: This method returns the text decoder associated with the instance.

        Raises:
            None.
        """
        return self.text_decoder

    def get_output_embeddings(self):
        """
        Method: get_output_embeddings

        Returns the output embeddings of the SeamlessM4TForSpeechToText model.

        Args:
            self: The instance of the SeamlessM4TForSpeechToText class.

        Returns:
            None

        Raises:
            None.

        Description:
            This method is used to retrieve the output embeddings of the SeamlessM4TForSpeechToText model.
            The output embeddings represent the final layer of the model, which encodes the input speech into a
            fixed-length vector representation. The output embeddings can be used for various downstream tasks
            such as speech-to-text conversion, speaker identification, or speech similarity analysis.

            Note that the output embeddings are specific to the SeamlessM4TForSpeechToText model and may not be
            compatible with other models or applications. The embeddings are not modified by this method and are
            provided as-is.

        Example:
            ```python
            >>> model = SeamlessM4TForSpeechToText()
            >>> output_embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets new embeddings for the output layer of the SeamlessM4TForSpeechToText class.

        Args:
            self (SeamlessM4TForSpeechToText): The instance of the SeamlessM4TForSpeechToText class.
            new_embeddings (object): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Method:
            get_input_embeddings

        Description:
            This method retrieves the input embeddings for the SeamlessM4TForSpeechToText class.

        Parameters:
            self: (SeamlessM4TForSpeechToText) The instance of the class.

        Returns:
            None.

        Raises:
            None

        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TForSpeechToText model.

        Args:
            self (SeamlessM4TForSpeechToText): The instance of the SeamlessM4TForSpeechToText class.
            value (object): The input embeddings to be set for the model. It should be of type 'object' and should
                contain the embedding tokens.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        """
        Tie weights of the SeamlessM4TForSpeechToText model.

        Args:
            self: SeamlessM4TForSpeechToText
                The instance of SeamlessM4TForSpeechToText class.

        Returns:
            None: This method modifies the weights of the model in place.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

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
        '''
        This method constructs a SeamlessM4TForSpeechToText model for speech to text conversion.

        Args:
            self (SeamlessM4TForSpeechToText): The instance of the SeamlessM4TForSpeechToText class.
            input_features (mindspore.Tensor, optional): The input features for the model. Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask for the input. Default is None.
            decoder_input_ids (mindspore.Tensor, optional): The input IDs for the decoder. Default is None.
            decoder_attention_mask (mindspore.Tensor, optional): The attention mask for the decoder. Default is None.
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The output of the encoder. Default is None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values for the model. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The embedded inputs for the model. Default is None.
            decoder_inputs_embeds (mindspore.Tensor, optional): The embedded inputs for the decoder. Default is None.
            labels (mindspore.Tensor, optional): The labels for the model. Default is None.
            use_cache (bool, optional): Indicates whether to use cache. Default is None.
            output_attentions (bool, optional): Indicates whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Indicates whether to output hidden states. Default is None.
            return_dict (bool, optional): Indicates whether to return a dictionary. Default is None.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The output of the model which can be either Seq2SeqLMOutput
                or a tuple of mindspore.Tensor.

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
            [`~utils.ModelOutput`] or `mindspore.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`. The possible
                [`~utils.ModelOutput`] types are:

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
        This method prepares inputs for generation in the SeamlessM4TForSpeechToText class.

        Args:
            self: The instance of the class.
            decoder_input_ids (Tensor): The input ids for the decoder.
                It is a tensor containing the input sequence tokens.
            past_key_values (tuple, optional): The past key values for autoregressive generation.
                It is a tuple containing the past key and value tensors.
            attention_mask (Tensor, optional): The attention mask for the input.
                It is a tensor containing the attention mask values.
            use_cache (bool, optional): Whether to use caching for the computation.
            encoder_outputs (tuple, optional): The outputs from the encoder.
                It is a tuple containing the encoder output tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (None): The input ids, which are set to None.
                - 'encoder_outputs' (Tensor): The encoder outputs to be used in the generation process.
                - 'past_key_values' (tuple): The past key values for autoregressive generation.
                - 'decoder_input_ids' (Tensor): The modified decoder input ids for the generation process.
                - 'attention_mask' (Tensor, optional): The attention mask for the input.
                - 'use_cache' (bool, optional): Whether to use caching for the computation.

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
        _reorder_cache method in the SeamlessM4TForSpeechToText class.

        This method reorders the past key values based on the given beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each element of the tuple is a tuple representing the past key values for a layer.
                The past key values are used for caching and are expected to be in the format
                (key, value, attention_mask).
            beam_idx (tensor): A tensor containing the indices of the beams to reorder the past key values.
                The tensor should be of type long and of shape (batch_size,).

        Returns:
            None.

        Raises:
            IndexError: If the beam_idx tensor is out of bounds for the past_key_values.
            ValueError: If the past_key_values or beam_idx parameters are not in the expected format or type.

        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):

    """
    This class represents a SeamlessM4T model for text-to-speech conversion.
    It is a subclass of the SeamlessM4TPreTrainedModel class.

    The class includes various methods for generating translated audio waveforms based on input text.
    It utilizes a text encoder, text decoder, and LM head to convert input text into speech.

    Methods:
        __init__: Initializes the SeamlessM4TForTextToSpeech model.
        get_encoder: Returns the text encoder.
        get_decoder: Returns the text decoder.
        get_output_embeddings: Returns the LM head for output embeddings.
        set_output_embeddings: Sets the LM head for output embeddings to the provided new embeddings.
        get_input_embeddings: Returns the input embeddings for the text decoder.
        set_input_embeddings: Sets the input embeddings for the text encoder and decoder to the provided value.
        _tie_weights: Ties the weights of the text encoder, decoder, and LM head if specified in the configuration.
        construct: Constructs the SeamlessM4T model for text-to-speech conversion.
        generate: Generates translated audio waveforms based on input text.
        prepare_inputs_for_generation: Prepares inputs for generation during text-to-speech conversion.
        _reorder_cache: Reorders the cache for beam search during generation.

    Please refer to the method docstrings for more detailed information on each method's input and output parameters.
    """
    _keys_to_ignore_on_load_missing = ["speech_encoder"]
    main_input_name = "input_ids"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config: SeamlessM4TConfig):
        """
        Initializes the SeamlessM4TForTextToSpeech class.

        Args:
            self: The instance of the class.
            config (SeamlessM4TConfig): An instance of SeamlessM4TConfig containing the configuration parameters
                for the model. It specifies the vocab_size, hidden_size, and pad_token_id for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        """
        This method 'get_encoder' is defined in the class 'SeamlessM4TForTextToSpeech' and is used to retrieve
        the text encoder.

        Args:
            self: The instance of the class.

        Returns:
            text_encoder: This method returns the text encoder associated with the instance.

        Raises:
            None.
        """
        return self.text_encoder

    def get_decoder(self):
        """
        Method to retrieve the text decoder for the SeamlessM4TForTextToSpeech class.

        Args:
            self (SeamlessM4TForTextToSpeech): The instance of the SeamlessM4TForTextToSpeech class.
                This parameter is required to access the text decoder specific to the instance.

        Returns:
            text_decoder: This method returns the text decoder associated with the instance of
                SeamlessM4TForTextToSpeech. The text decoder is used to decode text data into a suitable format
                for text-to-speech conversion.

        Raises:
            None.
        """
        return self.text_decoder

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the SeamlessM4TForTextToSpeech model.

        Args:
            self: An instance of the SeamlessM4TForTextToSpeech class.

        Returns:
            lm_head: The method returns the output embeddings of the model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the SeamlessM4TForTextToSpeech class.

        Args:
            self (SeamlessM4TForTextToSpeech): The instance of the SeamlessM4TForTextToSpeech class.
            new_embeddings (object): The new output embeddings to be set. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings for the SeamlessM4TForTextToSpeech class.

        Args:
            self (SeamlessM4TForTextToSpeech): An instance of the SeamlessM4TForTextToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TForTextToSpeech model.

        Args:
            self (SeamlessM4TForTextToSpeech): The instance of the SeamlessM4TForTextToSpeech class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, embed_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        """
        Tie word embeddings and language model head weights in the SeamlessM4TForTextToSpeech class.

        This method ties the weights of word embeddings and the language model head in the SeamlessM4TForTextToSpeech
        class if the 'tie_word_embeddings' flag is set to True. Tying weights means that the parameters of the
        specified modules will be shared, resulting in a reduced number of parameters in the model.

        Args:
            self (SeamlessM4TForTextToSpeech): An instance of the SeamlessM4TForTextToSpeech class.

        Returns:
            None.

        Raises:
            None.
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
    ) -> Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]:
        """
        Constructs the output of the 'SeamlessM4TForTextToSpeech' class.

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor, optional): The input tensor IDs. Default: None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Default: None.
            decoder_input_ids (mindspore.Tensor, optional): The decoder input tensor IDs. Default: None.
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask tensor. Default: None.
            encoder_outputs (Tuple[Tuple[mindspore.Tensor]], optional): The encoder outputs. Default: None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values. Default: None.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings tensor. Default: None.
            decoder_inputs_embeds (mindspore.Tensor, optional): The decoder input embeddings tensor. Default: None.
            labels (mindspore.Tensor, optional): The labels tensor. Default: None.
            use_cache (bool, optional): Indicates whether to use cache. Default: None.
            output_attentions (bool, optional): Indicates whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Indicates whether to output hidden states. Default: None.
            return_dict (bool, optional): Indicates whether to use return dict. Default: None.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The output of the 'construct' method.

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
                "This is the same forward method as `SeamlessM4TForTextToText`."
                "It doesn't use the text-to-unit model `SeamlessM4TTextToUnitForConditionalGeneration`."
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
        spkr_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4TGenerationOutput]:
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
            spkr_id (`int`, *optional*, defaults to 0):
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
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:

                - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
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
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
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

        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + ops.arange(batch_size) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = mindspore.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size)
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.copy()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        spkr_id = mindspore.tensor([[spkr_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

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
        This method prepares inputs for generation in the SeamlessM4TForTextToSpeech class.

        Args:
            self (object): The instance of the class.
            decoder_input_ids (Tensor): The input tensor for the decoder. It represents the input ids for the
                decoder model.
            past_key_values (tuple, optional): The past key values for the model's self-attention layers.
                Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor. It masks the attention to prevent attending
                to padding tokens. Defaults to None.
            use_cache (bool, optional): Indicates whether to use the cache for fast decoding. Defaults to None.
            encoder_outputs (Tensor, optional): The output tensor from the encoder model. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared inputs for generation.
                The dictionary includes the following keys:

                - 'input_ids' (None): Represents the input ids for the model.
                - 'encoder_outputs' (Tensor): The output tensor from the encoder model.
                - 'past_key_values' (tuple): The past key values for the model's self-attention layers.
                - 'decoder_input_ids' (Tensor): The input tensor for the decoder.
                - 'attention_mask' (Tensor): The attention mask tensor.
                - 'use_cache' (bool): Indicates whether to use the cache for fast decoding.

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
        Reorders the cache for a given beam index in the SeamlessM4TForTextToSpeech class.

        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
                Each layer's past key-value state is represented as a tuple containing:

                - past_state (Tensor): The past state tensor for the current layer.
                - present_state (Tensor): The present state tensor for the current layer.
                - additional_state (Any): Additional state information for the current layer.

                The length of past_key_values corresponds to the number of layers in the model.
            beam_idx (Tensor): The beam index indicating the order in which to reorder the cache.
                It is used to select the past state from each layer's past key-value state tensor.

        Returns:
            tuple: The reordered past key-value states for each layer.
                The reordered past key-value state for each layer is represented as a tuple containing:

                - reordered_past_state (Tensor): The reordered past state tensor for the current layer.
                - reordered_present_state (Tensor): The reordered present state tensor for the current layer.
                - additional_state (Any): Additional state information for the current layer.

                The length of the returned tuple corresponds to the number of layers in the model.

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


class SeamlessM4TForSpeechToSpeech(SeamlessM4TPreTrainedModel):

    """
    The `SeamlessM4TForSpeechToSpeech` class is a subclass of `SeamlessM4TPreTrainedModel` that represents a
    speech-to-speech translation model. It is designed to convert speech in one language to speech in another language.

    Methods:
        `__init__`: Initializes the `SeamlessM4TForSpeechToSpeech` class.
        `get_encoder`: Returns the speech encoder.
        `get_decoder`: Returns the text decoder.
        `get_output_embeddings`: Returns the output embeddings.
        `set_output_embeddings`: Sets the output embeddings to the given new embeddings.
        `get_input_embeddings`: Returns the input embeddings.
        `set_input_embeddings`: Sets the input embeddings to the given value.
        `_tie_weights`: Ties the weights of the text decoder embeddings and the shared embeddings
            if `tie_word_embeddings` is set to `True` in the configuration.
        `construct`: Constructs the speech-to-speech translation model and returns the output.
        `generate`: Generates translated audio waveforms.
        `_reorder_cache`: Reorders the past key values for generation.

    Please refer to the code for more detailed information on each method.
    """
    _keys_to_ignore_on_load_missing = ["text_encoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config):
        """
        Initializes the SeamlessM4TForSpeechToSpeech class.

        Args:
            self: An instance of the SeamlessM4TForSpeechToSpeech class.
            config: A configuration object containing various settings for the model.
                It must have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden state.
                - pad_token_id (int): The ID of the padding token.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        """
        Returns the speech encoder used by the SeamlessM4TForSpeechToSpeech class.

        Args:
            self: An instance of the SeamlessM4TForSpeechToSpeech class.

        Returns:
            None

        Raises:
            None
        """
        return self.speech_encoder

    def get_decoder(self):
        """
        Returns the text decoder used by the SeamlessM4TForSpeechToSpeech class.

        Args:
            self (SeamlessM4TForSpeechToSpeech):
                An instance of the SeamlessM4TForSpeechToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder

    def get_output_embeddings(self):
        """
        This method returns the output embeddings of the SeamlessM4TForSpeechToSpeech instance.

        Args:
            self: SeamlessM4TForSpeechToSpeech - The instance of the SeamlessM4TForSpeechToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the new output embeddings for the SeamlessM4TForSpeechToSpeech model.

        Args:
            self (SeamlessM4TForSpeechToSpeech): The instance of the SeamlessM4TForSpeechToSpeech class.
            new_embeddings (object): The new output embeddings to be set for the model. It can be of any valid type.

        Returns:
            None.

        Raises:
            None:
                However, if the new_embeddings parameter is not of a compatible type, it may raise a TypeError.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the SeamlessM4TForSpeechToSpeech model.

        Args:
            self (SeamlessM4TForSpeechToSpeech): The instance of the SeamlessM4TForSpeechToSpeech class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the SeamlessM4TForSpeechToSpeech model.

        Args:
            self (SeamlessM4TForSpeechToSpeech): The instance of the SeamlessM4TForSpeechToSpeech class.
            value: The input embeddings to be set for the model. It should be a tensor or any compatible type.

        Returns:
            None: This method modifies the input embeddings for the model in place.

        Raises:
            No specific exceptions are documented for this method. However, potential exceptions could include
            TypeError if the input value is not compatible with the model's requirements.
        """
        self.text_decoder.embed_tokens = value

    def _tie_weights(self):
        """
        Method to tie weights of specified layers in the SeamlessM4TForSpeechToSpeech class.

        Args:
            self (SeamlessM4TForSpeechToSpeech): The instance of the SeamlessM4TForSpeechToSpeech class.
                Used to access the configuration parameters and layers needed for tying weights.

        Returns:
            None: This method modifies the weights of specified layers in-place.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

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
        Method 'construct' in the class 'SeamlessM4TForSpeechToSpeech'.

        This method constructs a sequence-to-sequence model for speech-to-speech translation.

        Args:
            self: The object instance.
            input_features (mindspore.Tensor): Input features for the speech encoder.
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on padding tokens.
            decoder_input_ids (Optional[mindspore.Tensor]): Input IDs for the decoder.
            decoder_attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on padding tokens
                in the decoder.
            encoder_outputs (Optional[Tuple[Tuple[mindspore.Tensor]]]): Output states of the encoder.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key values for caching in the decoder.
            inputs_embeds (Optional[mindspore.Tensor]): Embedded inputs for the encoder.
            decoder_inputs_embeds (Optional[mindspore.Tensor]): Embedded inputs for the decoder.
            labels (Optional[mindspore.Tensor]): Labels for training.
            use_cache (Optional[bool]): Flag to indicate whether to use caching.
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions.
            output_hidden_states (Optional[bool]): Flag to indicate whether to output hidden states.
            return_dict (Optional[bool]): Flag to indicate whether to return a dictionary of outputs.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[mindspore.Tensor]]: The constructed sequence-to-sequence model output.

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
                "This is the same forward method as `SeamlessM4TForSpeechToText`. It doesn't use `self.t2u_model`."
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
        spkr_id: Optional[int] = 0,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4TGenerationOutput]:
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
            spkr_id (`int`, *optional*, defaults to 0):
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
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:

                - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
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
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
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

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + ops.arange(batch_size) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = mindspore.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size)
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.copy()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        spkr_id = mindspore.tensor([[spkr_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the given beam index.

        Args:
            past_key_values (tuple): A tuple of past key values for each layer in the model.
            beam_idx (int): The index of the beam to reorder the cache for.

        Returns:
            None: The method updates the order of the cache in place.

        Raises:
            ValueError: If the beam index is out of range or invalid.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

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
        Prepare the inputs for generation in the SeamlessM4TForSpeechToSpeech class.

        Args:
            self (SeamlessM4TForSpeechToSpeech): The instance of the SeamlessM4TForSpeechToSpeech class.
            decoder_input_ids (torch.Tensor): The input tensor for the decoder. Shape: (batch_size, sequence_length)
            past_key_values (tuple or None): The cached key-value pairs of the past decoder states. Default: None
            attention_mask (torch.Tensor or None): The attention mask tensor. Shape: (batch_size, sequence_length)
            use_cache (bool or None): Whether to use caching for the decoder. Default: None
            encoder_outputs (tuple or None): The outputs of the encoder. Default: None

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation.

                - 'input_ids' (None): Placeholder for input IDs.
                - 'encoder_outputs' (tuple or None): The outputs of the encoder.
                - 'past_key_values' (tuple or None): The cached key-value pairs of the past decoder states.
                - 'decoder_input_ids' (torch.Tensor): The updated input tensor for the decoder.
                - 'attention_mask' (torch.Tensor or None): The attention mask tensor.
                - 'use_cache' (bool or None): Whether to use caching for the decoder.

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


class SeamlessM4TModel(SeamlessM4TPreTrainedModel):

    """
    SeamlessM4TModel represents a model for seamless multimodal translation and synthesis tasks.
    It provides methods for initializing the model, setting modality, retrieving encoders, handling embeddings,
    constructing the model, generating translations and audio waveforms, and preparing inputs for generation.

    Attributes:
        config: An object containing the configuration parameters for the model.

    Methods:
        __init__: Initializes an instance of the SeamlessM4TModel class.
        set_modality: Sets the modality for the SeamlessM4TModel instance.
        get_encoder: Returns the appropriate encoder based on the current modality.
        get_output_embeddings: Returns the output embeddings of the SeamlessM4TModel.
        set_output_embeddings: Sets the output embeddings of the SeamlessM4TModel.
        get_input_embeddings: Get the input embeddings for the SeamlessM4TModel.
        set_input_embeddings: Sets the input embeddings for the SeamlessM4TModel.
        _tie_weights: Ties the weights of specified layers in the SeamlessM4TModel.
        construct: Constructs the SeamlessM4TModel.
        generate: Generates translated token ids and/or translated audio waveforms.
        prepare_inputs_for_generation: Prepares inputs for generation.
        _reorder_cache: Reorders the cache of past key values for the SeamlessM4TModel.
    """
    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    def __init__(self, config, current_modality="text"):
        """
        Initializes an instance of the SeamlessM4TModel class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration parameters for the model.
            current_modality (str, optional): The current modality to be used. Defaults to 'text'.
                Valid values are 'text' and 'speech'.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.speech_encoder = SeamlessM4TSpeechEncoder(config)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.current_modality = current_modality
        if current_modality == "speech":
            self.main_input_name = "input_features"

        # these models already call post_init in their initialization
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def set_modality(self, modality="text"):
        """
        This method sets the modality for the SeamlessM4TModel instance.

        Args:
            self (SeamlessM4TModel): The instance of SeamlessM4TModel.
            modality (str): The modality to be set. It must be either 'text' or 'speech'.

        Returns:
            None.

        Raises:
            ValueError: If the provided modality is not valid i.e., not 'text' or 'speech'.
        """
        if modality == "text":
            self.main_input_name = "input_ids"
            self.current_modality = "text"
        elif modality == "speech":
            self.main_input_name = "input_features"
            self.current_modality = "speech"
        else:
            raise ValueError(f"`modality={modality}` is not a valid modality. It must be `text` or `speech`.")

    def get_encoder(self):
        """
        Returns the appropriate encoder based on the current modality.

        Args:
            self: An instance of the SeamlessM4TModel class.

        Returns:
            encoder:
                The encoder object corresponding to the current modality. If the current modality is 'text',
                the method returns the text_encoder. Otherwise, it returns the speech_encoder.

        Raises:
            None.

        Note:
            The current_modality attribute must be set before calling this method, otherwise it will return None.
        """
        if self.current_modality == "text":
            return self.text_encoder
        return self.speech_encoder

    def get_output_embeddings(self):
        """
        This method returns the output embeddings of the SeamlessM4TModel.

        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel class.

        Returns:
            None: This method returns the output embeddings of the SeamlessM4TModel as a value of type None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the SeamlessM4TModel.

        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel class.
            new_embeddings (object): The new output embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        """
        Get the input embeddings for the SeamlessM4TModel.

        Args:
            self: An instance of the SeamlessM4TModel class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the text decoder of the SeamlessM4TModel.
        The input embeddings are used as the initial input for the model's text decoding process. The embeddings are
        obtained by calling the 'embed_tokens' method of the text decoder. The 'embed_tokens' method maps the input
        tokens to their corresponding embeddings, which are then used as input for the model.

        No exceptions are raised by this method.
        """
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the SeamlessM4TModel.

        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel.
            value (object): The input embeddings to be set for the model.
                It should be an object representing the input embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        """
        This method ties the weights of specified layers in the SeamlessM4TModel.

        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

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
        """
        Constructs the SeamlessM4TModel.

        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel.
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
            Warning: If `use_cache` is True and `labels` is provided, the `use_cache` argument is changed to False.
            Warning: If `decoder_input_ids` and `decoder_inputs_embeds` are None and `labels` is provided,
                the `decoder_input_ids` is set to shifted `labels`.
            Warning: If `input_features` is not None and `input_ids` is not None, `input_features` will be used in
                priority through the `speech_encoder`. Make sure that `input_features` and `input_ids` are
                mutually exclusive.
            Warning: If `inputs_embeds` is not None and `input_features` is not None, `input_features` will be used in
                priority through `speech_encoder`. `inputs_embeds` will be ignored.
            Warning: If the current modality is 'speech' and `attention_mask` is not None, `sub_sampled_lengths` will
                be computed from `attention_mask`.
            Warning: If the current modality is 'speech' and `attention_mask` is not None, `encoder_attention_mask` will
                be computed using `hidden_states` and `seq_lens`.
            Warning: If the current modality is 'text', `encoder_outputs` will be computed using `input_ids`,
                `attention_mask`, `inputs_embeds`, `output_attentions`, `output_hidden_states`, and `return_dict`.
            Warning: If `encoder_outputs` is not an instance of BaseModelOutput and `return_dict` is True,
                `encoder_outputs` will be converted to a BaseModelOutput.
            Warning: If `labels` is not None, the `masked_lm_loss` is computed using `lm_logits` and `labels`.
            Warning: If not `return_dict`, the `outputs` will be a combination of `decoder_outputs` and
                `encoder_outputs`.
            Warning: If `return_dict` is False and `masked_lm_loss` is not None, the `output` will be a combination
                of `lm_logits` and `outputs`.
            Warning: If `return_dict` is True, the `output` will be a Seq2SeqLMOutput including `masked_lm_loss`,
                `lm_logits`, `past_key_values`, `decoder_hidden_states`, `decoder_attentions`, `cross_attentions`,
                `encoder_last_hidden_state`, `encoder_hidden_states`, and `encoder_attentions`.
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
                "This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`"
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
                "This calls the same method `forward` as `SeamlessM4TForTextToText` and `SeamlessM4TForSpeechToText`"
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
        spkr_id: Optional[int] = 0,
        generate_speech: Optional[bool] = True,
        **kwargs,
    ) -> Union[mindspore.Tensor, SeamlessM4TGenerationOutput]:
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
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.

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
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor], ModelOutput]`:

                - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
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
                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports
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

        # take care of num_return_sequences
        # take most probable hidden states per batch of return_sequences
        # (batch_size*num_return_sequences, ...) -> (batch_size,...)
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = (
                idx_most_probable_sequences_per_batch + ops.arange(batch_size) * num_return_sequences
            )
            sequences = sequences[idx_most_probable_sequences_per_batch]

        # get decoder last hidden state - must do a pass through the text decoder
        t2u_input_embeds = self.text_decoder(
            input_ids=sequences,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        pad_token_id = self.generation_config.pad_token_id

        # Compute new attention mask
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech["attention_mask"] = t2u_model_attention_mask

        # Compute t2u decoder_input_ids
        t2u_decoder_input_ids = kwargs_speech.get("decoder_input_ids")
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = mindspore.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size)
        kwargs_speech["decoder_input_ids"] = t2u_decoder_input_ids

        # second generation
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.copy()

        # get rid of t2u_decoder_input_ids
        unit_ids = unit_ids[:, kwargs_speech["decoder_input_ids"].shape[1] :]
        # replace eos per pad
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        # offset of control symbols
        unit_ids = ops.where(
            unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset
        )

        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = mindspore.tensor([[vocoder_tgt_lang_id]] * len(unit_ids))

        spkr_id = mindspore.tensor([[spkr_id]] * len(unit_ids))

        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)

        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                sequences=sequences,
                unit_sequences=output_unit_ids,
            )

        return waveform, waveform_lengths

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
        
        This method takes 6 parameters: self, decoder_input_ids, past_key_values, attention_mask, use_cache,
        encoder_outputs.
        
        Args:
            self (SeamlessM4TModel): The instance of the SeamlessM4TModel class.
            decoder_input_ids (Tensor): The input tensor for the decoder.
                It represents the input IDs for the decoder model.
            past_key_values (Tuple, optional): The past key values for the transformer model. Default is None.
            attention_mask (Tensor, optional): The attention mask tensor. Default is None.
            use_cache (bool, optional): A flag indicating whether to use caching. Default is None.
            encoder_outputs (Tensor, optional): The output tensor from the encoder model. Default is None.
        
        Returns:
            dict: A dictionary containing the input IDs, encoder outputs, past key values, decoder input IDs,
                attention mask, and use_cache.
        
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
        Reorders the cache of past key values for the SeamlessM4TModel.
        
        Args:
            past_key_values (tuple): A tuple containing the past key values for the model's cache.
            beam_idx (Tensor): A tensor representing the beam index for reordering the cache.
        
        Returns:
            tuple: The reordered past key values.
        
        Raises:
            IndexError: If the beam index is out of range.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

__all__ = [
    "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",
    "SeamlessM4TForTextToSpeech",
    "SeamlessM4TForSpeechToSpeech",
    "SeamlessM4TForTextToText",
    "SeamlessM4TForSpeechToText",
    "SeamlessM4TModel",
    "SeamlessM4TPreTrainedModel",
    "SeamlessM4TCodeHifiGan",
    "SeamlessM4THifiGan",
    "SeamlessM4TTextToUnitForConditionalGeneration",
    "SeamlessM4TTextToUnitModel",
]
