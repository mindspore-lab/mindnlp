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
""" MindSpore Hubert model. """

from typing import Optional, Tuple, Union

import numpy as np

import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal, Uniform, HeNormal

from mindnlp.modules.functional.weight_norm import weight_norm
from mindnlp.modules.functional import finfo

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ....utils import logging

from .configuration_hubert import HubertConfig

__all__ = [
    'HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST',
    'HubertPreTrainedModel',
    'HubertModel',
    'HubertForCTC',
    'HubertForSequenceClassification',
]

logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 1

HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hubert-base-ls960",
    # See all Hubert models at https://hf-mirror.com/models?filter=hubert
]


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[Tensor] = None,
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
        attention_mask.sum(-1).detach().tolist()
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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->Hubert
class HubertNoLayerNormConvLayer(nn.Cell):

    """
    HubertNoLayerNormConvLayer is a Python class representing a convolutional layer without layer normalization.
    This class inherits from nn.Cell.
    
    This class initializes with the following parameters:

    - config: A HubertConfig object containing configuration settings.
    - layer_id: An integer representing the layer identifier.

    The construct method applies a convolutional operation and an activation function to the input hidden states.

    Attributes:
        in_conv_dim: Integer representing the input convolutional dimension.
        out_conv_dim: Integer representing the output convolutional dimension.
        conv: nn.Conv1d object with parameters for the convolutional operation.
        activation: Activation function defined in the ACT2FN dictionary based on the config's feat_extract_activation setting.

    Methods:
        construct(hidden_states): Applies the convolutional operation and activation function to the input hidden_states.

    """
    def __init__(self, config: HubertConfig, layer_id=0):
        """
        Initializes a HubertNoLayerNormConvLayer.

        Args:
            self: The object itself.
            config (HubertConfig): The configuration object containing model hyperparameters.
            layer_id (int): The index of the convolution layer.

        Returns:
            None.

        Raises:
            ValueError: If layer_id is less than 0.
            KeyError: If the specified activation function in config is not found in the ACT2FN dictionary.
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            has_bias=config.conv_bias,
            pad_mode="valid",
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def construct(self, hidden_states):
        """
        Constructs the hidden states of the HubertNoLayerNormConvLayer.

        Args:
            self (HubertNoLayerNormConvLayer): An instance of the HubertNoLayerNormConvLayer class.
            hidden_states (Tensor): The input hidden states to be processed.
                Expected shape is (batch_size, channels, sequence_length).

        Returns:
            Tensor: The processed hidden states after applying the convolutional layer and activation function.
                The shape is (batch_size, channels, sequence_length).

        Raises:
            None.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->Hubert
class HubertLayerNormConvLayer(nn.Cell):

    """
    The HubertLayerNormConvLayer class represents a layer with convolution, layer normalization,
    and activation functions for the HuBERT model. It inherits from nn.Cell.

    This class initializes with a HubertConfig instance and a layer ID. It defines a convolutional layer with
    specified input and output dimensions, kernel size, stride, bias, and padding mode. It also applies layer
    normalization and an activation function to the input hidden states.

    The construct method takes hidden states as input, applies the convolution, layer normalization, and
    activation function, and returns the processed hidden states.
    """
    def __init__(self, config: HubertConfig, layer_id=0):
        """
        Initializes a new instance of the HubertLayerNormConvLayer class.

        Args:
            self: The object itself.
            config (HubertConfig): The configuration object for the Hubert model.
            layer_id (int, optional): The ID of the layer. Defaults to 0.

        Returns:
            None.

        Raises:
            ValueError: If the provided layer_id is less than 0.
            TypeError: If the provided config is not an instance of HubertConfig.
            KeyError: If the provided config does not contain required attributes.
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            has_bias=config.conv_bias,
            pad_mode="valid",
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim)
        self.activation = ACT2FN[config.feat_extract_activation]

    def construct(self, hidden_states):
        """
        This method constructs a HubertLayerNormConvLayer by applying convolution, layer normalization, and
        activation functions to the input hidden states.

        Args:
            self: The instance of the HubertLayerNormConvLayer class.
            hidden_states: A tensor representing the input hidden states that will undergo the transformation.
                It should have the shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not return any value directly.
                The hidden_states tensor is modified in place and returned after the transformations.

        Raises:
            ValueError: If the hidden_states tensor does not have the expected shape.
            RuntimeError: If any error occurs during the convolution, layer normalization, or activation operations.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->Hubert
class HubertGroupNormConvLayer(nn.Cell):

    """A class representing a Group Normalization Convolutional Layer in the Hubert model.

    This class inherits from nn.Cell and is used to define a single layer of the Hubert model.
    The layer consists of a 1-dimensional convolutional operation followed by group normalization,
    an activation function, and returns the output hidden states.

    Attributes:
        in_conv_dim (int): The dimension of the input to the convolutional layer.
        out_conv_dim (int): The dimension of the output from the convolutional layer.
        conv (nn.Conv1d): The 1-dimensional convolutional operation.
        activation (function): The activation function applied to the hidden states.
        layer_norm (nn.GroupNorm): The group normalization operation.

    Methods:
        construct(hidden_states): Applies the convolutional operation, group normalization,
            and activation function to the input hidden states and returns the output.

    """
    def __init__(self, config: HubertConfig, layer_id=0):
        """
        Initializes a HubertGroupNormConvLayer object.

        Args:
            self (HubertGroupNormConvLayer): The instance of the HubertGroupNormConvLayer class.
            config (HubertConfig): An instance of HubertConfig class containing configuration parameters.
            layer_id (int): The ID of the layer, defaults to 0. Used to access specific convolutional layer configuration.

        Returns:
            None.

        Raises:
            ValueError: If layer_id is less than 0.
            KeyError: If the specified feature extraction activation function is not found in the ACT2FN dictionary.
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            has_bias=config.conv_bias,
            pad_mode="valid",
        )
        self.activation = ACT2FN[config.feat_extract_activation]
        # NOTE: the naming is confusing, but let it be...
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def construct(self, hidden_states):
        """
        Construct a HubertGroupNormConvLayer by applying a series of operations on the input hidden states.

        Args:
            self (HubertGroupNormConvLayer): An instance of the HubertGroupNormConvLayer class.
            hidden_states (tensor): The input hidden states to be processed.
                Expected shape: (batch_size, channels, height, width).

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->Hubert
class HubertPositionalConvEmbedding(nn.Cell):

    """
    Represents a Positional Convolutional Embedding layer for the Hubert model.

    This class inherits from nn.Cell and is used to apply positional convolutional embeddings to input hidden states.
    The layer uses a convolutional neural network to process the input hidden states with configurable parameters
    such as kernel size, padding, activation function, and bias.

    Attributes:
        conv (nn.Conv1d): Convolutional layer for processing hidden states.
        padding (HubertSamePadLayer): Padding layer to ensure input dimensions match convolutional operations.
        activation (ACT2FN): Activation function to apply after convolution and padding.

    Methods:
        __init__: Initializes the Positional Convolutional Embedding layer with the specified configuration.
        construct: Constructs the positional convolutional embedding by applying convolution, padding,
            and activation functions to the input hidden states.

    Returns:
        The processed hidden states with positional convolutional embeddings applied.

    Note:
        This class is designed specifically for the Hubert model and should be used within the model architecture
        for optimal performance.
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes an instance of the HubertPositionalConvEmbedding class.

        Args:
            self: The instance of the class.
            config (HubertConfig): The configuration object containing the settings for Hubert model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            pad_mode='pad',
            padding=config.num_conv_pos_embeddings // 2,
            group=config.num_conv_pos_embedding_groups,
            has_bias=True,      # TODO: confirm this
        )
        self.conv = weight_norm(self.conv, name='weight', dim=2)
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def construct(self, hidden_states):
        """
        Constructs the HubertPositionalConvEmbedding.

        Args:
            self (HubertPositionalConvEmbedding): The instance of the HubertPositionalConvEmbedding class.
            hidden_states (numpy.ndarray):
                The input hidden states of shape (batch_size, sequence_length, hidden_size), where batch_size
                represents the number of input samples, sequence_length represents the length of the input sequence,
                and hidden_size represents the dimensionality of the hidden states.
                The hidden states are expected to be in the format (batch_size, hidden_size, sequence_length).

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->Hubert
class HubertSamePadLayer(nn.Cell):

    """
    This class represents a layer in the Hubert model that performs same padding on the input hidden states.

    The HubertSamePadLayer class is a subclass of the nn.Cell class and provides functionality to remove padding from
    the input hidden states if necessary. It is specifically designed for the Hubert model and is used to ensure that
    the input hidden states have the same length as the target sequence for further processing.

    Attributes:
        num_pad_remove (int): The number of padding elements to remove from the input hidden states.
            It is determined based on the number of convolutional positional embeddings. If the number is even,
            num_pad_remove is set to 1, otherwise it is set to 0.

    Methods:
        __init__(num_conv_pos_embeddings):
            Initializes a new instance of the HubertSamePadLayer class.

            Args:

            - num_conv_pos_embeddings (int): The number of convolutional positional embeddings.

        construct(hidden_states):
            Constructs the output hidden states by removing the padding elements if necessary.

            Args:

            - hidden_states (Tensor): The input hidden states to be processed.

            Returns:

            - Tensor: The processed hidden states with padding elements removed if necessary.
    """
    def __init__(self, num_conv_pos_embeddings):
        """
        Args:
            self (object): The instance of the class.
            num_conv_pos_embeddings (int): The number of convolutional position embeddings used in the layer.
                It is used to calculate the value of 'num_pad_remove' based on whether it is even or odd.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def construct(self, hidden_states):
        """
        Constructs the hidden states for the HubertSamePadLayer.

        Args:
            self (HubertSamePadLayer): An instance of the HubertSamePadLayer class.
            hidden_states (Tensor): The input hidden states to be processed.
                Expected shape is (batch_size, sequence_length, hidden_size).

        Returns:
            None.

        Raises:
            None.
        """
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->Hubert
class HubertFeatureEncoder(nn.Cell):
    """Construct the features from raw audio waveform"""
    def __init__(self, config: HubertConfig):
        """
        Initializes a new instance of HubertFeatureEncoder.

        Args:
            self: The instance of the class.
            config (HubertConfig): An instance of HubertConfig containing configuration parameters for the feature encoder.
                It specifies the normalization type to be used for feature extraction.

                - config.feat_extract_norm (str): Specifies the normalization type, should be either 'group' or 'layer'.

        Returns:
            None.

        Raises:
            ValueError: If the normalization type specified in config.feat_extract_norm is not 'group' or 'layer'.
        """
        super().__init__()
        if config.feat_extract_norm == "group":
            conv_layers = [HubertGroupNormConvLayer(config, layer_id=0)] + [
                HubertNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [HubertLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.CellList(conv_layers)
        self._requires_grad = True

    def _freeze_parameters(self):
        """
        Method _freeze_parameters in the class HubertFeatureEncoder freezes the parameters of the model
        by setting their 'requires_grad' attribute to False.

        Args:
            self (HubertFeatureEncoder): The instance of the HubertFeatureEncoder class.

        Returns:
            None.

        Raises:
            None.
        """
        for _, param in self.parameters_and_names():
            param.requires_grad = False
        self._requires_grad = False

    def construct(self, input_values):
        """
        Constructs the hidden states of the HubertFeatureEncoder.

        Args:
            self (HubertFeatureEncoder): An instance of the HubertFeatureEncoder class.
            input_values (array-like): The input values for constructing the hidden states.
                It should be a 2-dimensional array with shape (n_samples, n_features).

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class HubertFeatureProjection(nn.Cell):

    '''
    Represents a feature projection module for the Hubert model.

    This class inherits from nn.Cell and implements methods for initializing the feature projection layer
    and performing feature projection on hidden states.

    Attributes:
        feat_proj_layer_norm (bool): Indicates whether feature projection layer normalization is enabled.
        layer_norm (nn.LayerNorm): If feat_proj_layer_norm is True, this attribute represents the layer normalization module.
        projection (nn.Dense): The dense layer for feature projection.
        dropout (nn.Dropout): The dropout layer for feature projection.

    Methods:
        __init__: Initializes the feature projection layer with the given configuration.
        construct: Performs feature projection on the input hidden states and returns the projected hidden states.
    '''
    def __init__(self, config: HubertConfig):
        """
        Initializes a new instance of HubertFeatureProjection.

        Args:
            self: The instance of the HubertFeatureProjection class.
            config (HubertConfig):
                An instance of HubertConfig containing configuration parameters for the feature projection.

                - feat_proj_layer_norm (bool): Indicates whether layer normalization should be applied.
                - conv_dim (list): List of dimensions for convolutional layers.
                - layer_norm_eps (float): Epsilon value for layer normalization.
                - hidden_size (int): Size of the hidden layer.
                - feat_proj_dropout (float): Dropout rate for feature projection.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type HubertConfig.
            AttributeError: If the config object does not contain the required attributes.
            ValueError: If the config attributes are not within the specified range or format.
        """
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], epsilon=config.layer_norm_eps)
        self.projection = nn.Dense(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(p=config.feat_proj_dropout)

    def construct(self, hidden_states):
        """
        Constructs the feature projection for the HubertFeatureProjection class.

        Args:
            self: An instance of the HubertFeatureProjection class.
            hidden_states (Tensor): The input hidden states to be projected.
                It should have a shape of (batch_size, seq_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Hubert
class HubertAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[HubertConfig] = None,
    ):
        """
        Initializes a HubertAttention object.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads to use.
            dropout (float, optional): The dropout probability. Default is 0.0.
            is_decoder (bool, optional): Whether the attention mechanism is used in a decoder. Default is False.
            bias (bool, optional): Whether to include bias in the linear projections. Default is True.
            is_causal (bool, optional): Whether the attention is causal. Default is False.
            config (Optional[HubertConfig], optional): The configuration object for the attention mechanism.
                Default is None.

        Returns:
            None.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
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

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        """
        Method _shape in class HubertAttention.

        Args:
            self (HubertAttention): The instance of the HubertAttention class.
            tensor (Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None: This method reshapes the input tensor based on the provided sequence length and batch size.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
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
            # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
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
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}")
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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward with Wav2Vec2->Hubert
class HubertFeedForward(nn.Cell):

    """
    The HubertFeedForward class represents a feedforward neural network layer for the Hubert model.
    It inherits from nn.Cell and implements the feedforward computation for the hidden states.

    Attributes:
        config (HubertConfig): The configuration object for the Hubert model.

    Methods:
        __init__: Initializes the HubertFeedForward instance with the provided configuration.
        construct: Constructs the feedforward neural network layer using the provided hidden_states.

    Example:
        ```python
        Instantiate the HubertFeedForward class with a given configuration:
        >>> config = HubertConfig(...)
        >>> feed_forward_layer = HubertFeedForward(config)

        Perform the feedforward computation using the constructed layer:
        >>> hidden_states = ...
        >>> output = feed_forward_layer.construct(hidden_states)
        ```
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes the HubertFeedForward class with the specified configuration.

        Args:
            self: The instance of the HubertFeedForward class.
            config (HubertConfig):
                An instance of HubertConfig containing the configuration parameters for the feed-forward layer.
                The config parameter should be of type HubertConfig and is used to set up the feed-forward layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type HubertConfig.
        """
        super().__init__()
        self.intermediate_dropout = nn.Dropout(p=config.activation_dropout)
        self.intermediate_dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.output_dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=config.hidden_dropout)

    def construct(self, hidden_states):
        """
        Constructs the hidden states of the HubertFeedForward model.

        Args:
            self: An instance of the HubertFeedForward class.
            hidden_states (Tensor):
                The hidden states to be processed by the model.

                - Shape: (batch_size, sequence_length, hidden_size).
                - Purpose: Represents the input hidden states for the model.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer with Wav2Vec2->Hubert
class HubertEncoderLayer(nn.Cell):

    '''
    HubertEncoderLayer represents a single layer of the HubertEncoder.

    This class inherits from nn.Cell and contains methods to initialize the layer and construct the layer.
    The __init__ method initializes the layer with the given configuration, while the construct method applies the
    attention mechanism, dropout, layer normalization, feed forward, and final layer normalization to the input
    hidden states.

    Attributes:
        attention: An instance of HubertAttention representing the attention mechanism with specified parameters.
        dropout: An instance of nn.Dropout representing the dropout layer with a specified dropout rate.
        layer_norm: An instance of nn.LayerNorm representing the layer normalization with a specified epsilon.
        feed_forward: An instance of HubertFeedForward representing the feed forward layer with the given configuration.
        final_layer_norm: An instance of nn.LayerNorm representing the final layer normalization with a specified epsilon.

    Methods:
        __init__: Initializes the HubertEncoderLayer instance with the given configuration.
        construct: Applies the attention mechanism, dropout, layer normalization, feed forward, and final
        layer normalization to the input hidden states.

    '''
    def __init__(self, config: HubertConfig):
        """
        Initializes a HubertEncoderLayer instance.

        Args:
            self: The instance of HubertEncoderLayer.
            config (HubertConfig):
                An instance of HubertConfig containing configuration parameters.

                - config.hidden_size (int): The size of hidden layers.
                - config.num_attention_heads (int): The number of attention heads.
                - config.attention_dropout (float): The dropout probability for attention layers.
                - config.hidden_dropout (float): The dropout probability for hidden layers.
                - config.layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        """
        Method to construct the Hubert Encoder Layer.

        Args:
            self: Reference to the instance of the class.
            hidden_states (Tensor): Input hidden states to be processed.
            attention_mask (Tensor, optional): Mask to avoid attending over padding tokens. Default is None.
            output_attentions (bool, optional): Flag to indicate whether to output attention weights. Default is False.

        Returns:
            Tuple: A tuple containing the processed hidden states.
                If output_attentions is True, the tuple also includes the attention weights.

        Raises:
            None
        """
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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer with Wav2Vec2->Hubert
class HubertAttnAdapterLayer(nn.Cell):

    """
    Implements an adapter layer for attention modules in the Hubert model, optimizing training throughput
    by utilizing 3D tensor weights as parameters and bypassing the use of ModuleList.

    This class inherits from nn.Cell and provides functionality to construct adapter modules directly with 3D tensor
    weights as parameters, without using ModuleList, resulting in improved training throughput.

    Attributes:
        input_dim (int): The dimension of the input tensor for the adapter layer.
        hidden_dim (int): The hidden size dimension of the adapter layer.
        norm (LayerNorm): An instance of LayerNorm for normalizing the hidden states.
        linear_1 (Dense): An instance of Dense representing the first linear transformation.
        act_fn (ReLU): An instance of ReLU activation function.
        linear_2 (Dense): An instance of Dense representing the second linear transformation.

    Methods:
        construct:
            Constructs the adapter layer by applying normalization, linear transformations, and activation function
            to the given hidden_states tensor, and returns the resulting tensor.

    """
    def __init__(self, config: HubertConfig):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        super().__init__()
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.linear_1 = nn.Dense(self.hidden_dim, self.input_dim)
        self.act_fn = nn.ReLU()
        self.linear_2 = nn.Dense(self.input_dim, self.hidden_dim)

    def construct(self, hidden_states: Tensor):
        """
        Method to construct the attention adapter layer in the HubertAttnAdapterLayer class.

        Args:
            self (HubertAttnAdapterLayer): The instance of the HubertAttnAdapterLayer class.
            hidden_states (Tensor): The input hidden states tensor for the layer.

        Returns:
            hidden_states: The constructed hidden states after passing through the layer operations.

        Raises:
            None.
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayerStableLayerNorm with Wav2Vec2->Hubert
class HubertEncoderLayerStableLayerNorm(nn.Cell):

    """
    Represents a stable layer normalization encoder layer for the Hubert model.

    This class inherits from nn.Cell and contains methods for initializing the layer and constructing the layer
    with attention and feed-forward operations. It also includes an optional adapter layer.

    Attributes:
        attention: HubertAttention
            The attention mechanism for the encoder layer.
        dropout: nn.Dropout
            The dropout layer for the encoder layer.
        layer_norm: nn.LayerNorm
            The layer normalization for the encoder layer.
        feed_forward: HubertFeedForward
            The feed-forward network for the encoder layer.
        final_layer_norm: nn.LayerNorm
            The final layer normalization for the encoder layer.
        adapter_layer: HubertAttnAdapterLayer or None
            The optional adapter layer for the encoder layer.

    Methods:
        __init__:
            Initializes the encoder layer with the provided configuration.
        construct:
            Constructs the encoder layer with attention and feed-forward operations, and an optional adapter layer.

    Returns:
        outputs: Tuple[Tensor, ...]
            The outputs of the encoder layer, including hidden states and optionally attention weights.
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes a HubertEncoderLayerStableLayerNorm instance.

        Args:
            self: The instance of the HubertEncoderLayerStableLayerNorm class.
            config (HubertConfig):
                An instance of the HubertConfig class containing configuration parameters for the encoder layer.

                Parameters:

                - embed_dim (int): The dimension of the embedding.
                - num_heads (int): The number of attention heads.
                - dropout (float): The dropout probability for attention.
                - hidden_dropout (float): The dropout probability for hidden layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - adapter_attn_dim (int, optional): The dimension of the attention adapter layer. Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = HubertAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        This method constructs the Hubert encoder layer with stable layer normalization.

        Args:
            self: The instance of the HubertEncoderLayerStableLayerNorm class.
            hidden_states (Tensor): The input tensor representing the hidden states.
                This parameter is required for the construction of the encoder layer.
            attention_mask (Optional[Tensor]): An optional tensor representing the attention mask.
                It defaults to None and is used to mask padded tokens during attention computation.
            output_attentions (bool): A boolean flag indicating whether to output the attention weights.
                It defaults to False and is used to control whether the attention weights should be returned.

        Returns:
            tuple: A tuple containing the constructed hidden states. If output_attentions is True, the tuple also contains
                the attention weights.

        Raises:
            None
        """
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder with Wav2Vec2->Hubert
class HubertEncoder(nn.Cell):

    """
    A class representing the encoder component of the Hubert model.
    This class is responsible for processing input hidden states through multiple layers of HubertEncoderLayer.

    This class inherits from nn.Cell.

    Attributes:
        config (HubertConfig): The configuration object for the Hubert model.
        pos_conv_embed (HubertPositionalConvEmbedding):
            Instance of HubertPositionalConvEmbedding for positional convolutional embeddings.
        layer_norm (nn.LayerNorm): Layer normalization module.
        dropout (nn.Dropout): Dropout module for regularization.
        layers (nn.CellList): List of HubertEncoderLayer instances representing the encoder layers.

    Methods:
        construct(hidden_states, attention_mask, output_attentions, output_hidden_states, return_dict):
            Processes the input hidden states through the encoder layers and returns the final hidden states
            along with optional hidden states and attentions.

    Args:
        hidden_states (Tensor): The input hidden states to be processed by the encoder.
        attention_mask (Optional[Tensor]): Optional attention mask to mask out specific tokens during processing.
        output_attentions (bool): Flag indicating whether to output attention weights.
        output_hidden_states (bool): Flag indicating whether to output hidden states of each layer.
        return_dict (bool): Flag indicating whether to return the output as a BaseModelOutput dictionary.

    Returns:
        BaseModelOutput or tuple: A BaseModelOutput object containing the last hidden state, hidden states of all layers,
            and attention weights, or a tuple containing these elements based on the value of 'return_dict'.

    """
    def __init__(self, config: HubertConfig):
        """
        Initializes an instance of the HubertEncoder class.

        Args:
            self: The instance of the class.
            config (HubertConfig):
                The configuration object containing various settings for the HubertEncoder.

                - type: HubertConfig
                - purpose: It provides necessary configurations for initializing the encoder.
                - restrictions: Must be an instance of HubertConfig.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layers = nn.CellList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        This method constructs the Hubert encoder using the provided parameters and returns the final output.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The hidden states input tensor for the encoder.
            attention_mask (Optional[Tensor]):
                Optional attention mask tensor to mask certain elements in the hidden states. Default is None.
            output_attentions (bool): Flag indicating whether to output attentions. Default is False.
            output_hidden_states (bool): Flag indicating whether to output hidden states. Default is False.
            return_dict (bool): Flag indicating whether to return the output as a dictionary. Default is True.

        Returns:
            None: This method does not return any value explicitly,
                as it updates hidden states and hidden state-related outputs within the class instance.

        Raises:
            ValueError: If the attention mask dimensions are incompatible with the hidden states tensor.
            RuntimeError: If there is an issue during the execution of the encoder layers.
            TypeError: If the input types are incorrect or incompatible with the expected types.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * finfo(hidden_states.dtype, 'min')
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = ops.rand([])

            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm with Wav2Vec2->Hubert
class HubertEncoderStableLayerNorm(nn.Cell):

    """
    Class representing a Hubert encoder with stable layer normalization.

    This class implements an encoder model for the Hubert architecture with stable layer normalization.
    The encoder consists of multiple layers, each containing positional convolutional embeddings,
    layer normalization, and dropout, followed by a series of encoder layers.
    The encoder can process input hidden states, apply attention masks,
    and optionally output hidden states and self-attentions.

    Attributes:
        config (HubertConfig): The configuration object for the Hubert model.
        pos_conv_embed (HubertPositionalConvEmbedding): Positional convolutional embedding layer.
        layer_norm (nn.LayerNorm): Layer normalization for the hidden states.
        dropout (nn.Dropout): Dropout layer.
        layers (nn.CellList): List of encoder layers for processing the hidden states.

    Methods:
        construct(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
            Processes the input hidden states through the encoder layers.

            Args:

            - hidden_states (Tensor): Input hidden states to be processed.
            - attention_mask (Tensor, optional): Mask for attention scores. Defaults to None.
            - output_attentions (bool, optional): Whether to output self-attention matrices. Defaults to False.
            - output_hidden_states (bool, optional): Whether to output hidden states of each layer. Defaults to False.
            - return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to True.

            Returns:

            - BaseModelOutput: Object containing the last hidden state, hidden states of all layers, and self-attention matrices.
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes an instance of the HubertEncoderStableLayerNorm class.

        Args:
            self: The instance of the HubertEncoderStableLayerNorm class.
            config (HubertConfig): An instance of HubertConfig containing configuration parameters for the encoder.
                This parameter specifies the configuration settings for the encoder.
                It is a required parameter and must be of type HubertConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layers = nn.CellList([HubertEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Constructs the Hubert encoder stable layer norm.

        Args:
            self (HubertEncoderStableLayerNorm): The object instance.
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional):
                The attention mask of shape (batch_size, sequence_length) or (batch_size, 1, 1, sequence_length)
                indicating which tokens should be attended to. Defaults to None.
            output_attentions (bool, optional): Whether to return the attentions. Defaults to False.
            output_hidden_states (bool, optional): Whether to return the hidden states. Defaults to False.
            return_dict (bool, optional): Whether to return a dictionary instead of a BaseModelOutput. Defaults to True.

        Returns:
            None:
                This method does not return any value. It operates in place on the hidden_states
                and other internal buffers.

        Raises:
            ValueError: If the shapes of the input tensors are not compatible or if there are issues in the
                internal computations.
            RuntimeError: If there are errors during the computation or if the method is called in an invalid state.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * finfo(hidden_states.dtype, 'min')
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability =ops.rand([])

            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
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


class HubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = HubertConfig
    base_model_prefix = "hubert"
    main_input_name = "input_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Conv1d):
            cell.weight.set_data(initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
        if isinstance(cell, (nn.Dense, nn.Conv1d)) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def _get_feat_extract_output_lengths(self, input_lengths: Union[Tensor, int]):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/ops.nn.Conv1d.html
            #return ops.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: Tensor):
        """
        Method _get_feature_vector_attention_mask in class HubertPreTrainedModel.

        This method calculates the attention mask for feature vectors based on the provided feature_vector_length
        and attention_mask.

        Args:
            self: The instance of the class.
            feature_vector_length (int): The length of the feature vectors.
                This parameter specifies the length of the feature vectors to be used in calculating the attention mask.
                It must be a positive integer.
            attention_mask (Tensor): The attention mask tensor.
                This tensor indicates the positions of the actual tokens in the input sequence.
                It should be a 2D tensor with shape (batch_size, sequence_length) where batch_size is the number of sequences
                in the batch and sequence_length is the length of each sequence.

        Returns:
            None: This method does not return any value. It modifies the input attention_mask tensor in-place to generate the
                feature vector attention mask.

        Raises:
            ValueError: If feature_vector_length is not a positive integer or if attention_mask is not a valid tensor.
            IndexError: If there is an index out of range error during tensor operations.
            TypeError: If the data type of the attention_mask tensor is not supported for the operations in the method.
        """
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(mindspore.int64)
        batch_size = attention_mask.shape[0]

        attention_mask = ops.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(ops.arange(attention_mask.shape[0]), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


class HubertModel(HubertPreTrainedModel):

    """
    A class representing a Hubert model for speech recognition tasks.

    This class implements a Hubert model for processing speech input and generating relevant outputs.
    It includes methods for initializing the model, masking hidden states according to SpecAugment, and
    constructing the model's forward pass. The model utilizes a feature extractor, feature projection, and an encoder
    for processing input data and generating output representations.

    Attributes:
        config: HubertConfig
        feature_extractor: HubertFeatureEncoder
        feature_projection: HubertFeatureProjection
        encoder: HubertEncoder or HubertEncoderStableLayerNorm based on configuration

    Methods:
        __init__: Initializes the HubertModel with the provided configuration.
        _mask_hidden_states: Masks hidden states along the time and/or feature axes based on SpecAugment.
        construct: Constructs the forward pass of the model, processing input values and returning relevant outputs.

    Example:
        ```python
        >>> from transformers import AutoProcessor, HubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf
        ...
        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        ...
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)
        ...
        >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes the HubertModel with the provided configuration.

        Args:
            self: The instance of the HubertModel class.
            config (HubertConfig):
                An instance of the HubertConfig class representing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = Parameter(initializer(Uniform(), (config.hidden_size,), dtype=mindspore.float32))

        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
    def _mask_hidden_states(
        self,
        hidden_states: Tensor,
        mask_time_indices: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
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
            mask_time_indices = Tensor(mask_time_indices, dtype=mindspore.bool_)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = Tensor(mask_feature_indices, dtype=mindspore.bool_)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def construct(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        mask_time_indices: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """

        Returns:
            Union[Tuple, BaseModelOutput]

        Example:
            ```python
            >>> from transformers import AutoProcessor, HubertModel
            >>> from datasets import load_dataset
            >>> import soundfile as sf
            ...
            >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
            >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            ...
            ...
            >>> def map_to_array(batch):
            ...     speech, _ = sf.read(batch["file"])
            ...     batch["speech"] = speech
            ...     return batch
            ...
            ...
            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)
            ...
            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> hidden_states = model(input_values).last_hidden_state
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.swapaxes(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC with Wav2Vec2->Hubert, wav2vec2->hubert, WAV_2_VEC_2->HUBERT
class HubertForCTC(HubertPreTrainedModel):

    """
    A class representing the Hubert model for Connectionist Temporal Classification (CTC).

    This class extends the HubertPreTrainedModel class and provides additional methods for freezing the feature encoder
    and base model, as well as constructing the model and computing the CTC loss.

    Attributes:
        hubert (HubertModel): The Hubert model for feature extraction.
        dropout (Dropout): Dropout layer for regularization.
        target_lang (str): The target language for the model.
        lm_head (Dense): Fully connected layer for language modeling.

    Methods:
        __init__: Initializes the HubertForCTC instance with a given configuration and target language.
        tie_weights: Overwrites the tie_weights method to correctly load adapter weights when passing target_lang
            to from_pretrained().
        freeze_feature_encoder: Disables gradient computation for the feature encoder to prevent parameter updates
            during training.
        freeze_base_model: Disables gradient computation for the base model to prevent parameter updates during training.
        construct: Constructs the model and computes the CTC loss.

    Note:
        - The target_lang parameter is used for loading adapter weights and should not be passed
        if config.adapter_attn_dim is not defined.
        - The construct method computes the CTC loss for connectionist temporal classification tasks.

    Raises:
        ValueError: If the config.vocab_size is not defined when instantiating the model.
        ValueError: If target_lang is passed without config.adapter_attn_dim being defined.

    This class is intended to be used as a language model for CTC tasks, where labels are provided for training and
    the model outputs logits for each input sequence.
    """
    def __init__(self, config: HubertConfig, target_lang: Optional[str] = None):
        """
        Initializes a new instance of the HubertForCTC class.

        Args:
            self: The object instance.
            config (HubertConfig): The configuration object for the Hubert model.
            target_lang (Optional[str], default=None): The target language for the model.
                If specified, the model will be trained for the specified language.

        Returns:
            None

        Raises:
            ValueError: If the configuration does not define the vocabulary size of the language model head.

        This method initializes the HubertForCTC class by setting up the following components:

        - config: The configuration object for the Hubert model.
        - hubert: The HubertModel instance based on the provided configuration.
        - dropout: A dropout layer with the dropout probability defined in the configuration.
        - target_lang: The target language for the model, if specified.
        - lm_head: A dense layer with the output hidden size and vocabulary size defined in the configuration.

        Note:
            If the configuration has the 'add_adapter' attribute and it is set to True, the output hidden size will be
            the value of 'output_hidden_size'. Otherwise, it will be the value of 'hidden_size'.

        After initializing these components, the 'post_init' method is called to perform any additional setup tasks.
        """
        super().__init__(config)

        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(p=config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Dense(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Hubert so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Hubert never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config_class, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        if target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang)

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.hubert.parameters_and_names():
            param.requires_grad = False

    def construct(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
                the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
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
            labels = labels.astype(mindspore.int32)
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else ops.ones_like(input_values, dtype=mindspore.int64)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(mindspore.int64)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = ops.log_softmax(logits, axis=-1).swapaxes(0, 1)
            loss, log_alpha = ops.ctc_loss(
                log_probs,   # [T, N/B, C/NC]
                labels,      # [N/B, S], replace `flattened_targets`
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


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification with Wav2Vec2->Hubert, wav2vec2->hubert, WAV_2_VEC_2->HUBERT
class HubertForSequenceClassification(HubertPreTrainedModel):

    """
    HubertForSequenceClassification is a class that represents a sequence classification model based on the Hubert architecture.
    This class extends the HubertPreTrainedModel and provides functionality for sequence classification tasks.

    Methods:
        __init__: Initializes the sequence classification model with the provided configuration.
        freeze_feature_encoder:
            Disables gradient computation for the feature encoder to prevent parameter updates during training.
        freeze_base_model:
            Disables gradient computation for the base model parameters,
            allowing only the classification head to be updated.
        construct:
            Constructs the sequence classification model and computes the loss based on the
            provided input values and labels.

    Attributes:
        hubert: HubertModel instance for the sequence classification model.
        projector: nn.Dense layer for projecting hidden states to the classifier projection size.
        classifier: nn.Dense layer for classification predictions.
        layer_weights: Parameter for weighted layer sum computation.

    Note:
        - The class assumes a specific structure and functionality based on the provided code snippets.
    """
    def __init__(self, config: HubertConfig):
        """
        Initializes a new instance of HubertForSequenceClassification.

        Args:
            self: The instance of the class.
            config (HubertConfig): The configuration object for the Hubert model.

        Returns:
            None.

        Raises:
            ValueError: Raised if the 'config' object has the attribute 'add_adapter' set to True,
            as sequence classification does not support the use of Hubert adapters in this context.
        """
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError("Sequence classification does not support the use of Hubert adapters (config.add_adapter=True)")
        self.hubert = HubertModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = Parameter(ops.ones(num_layers) / num_layers)
        self.projector = nn.Dense(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Dense(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.hubert.parameters_and_names():
            param.requires_grad = False

    def construct(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(
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
            labels = labels.astype(mindspore.int32)
            loss = ops.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
