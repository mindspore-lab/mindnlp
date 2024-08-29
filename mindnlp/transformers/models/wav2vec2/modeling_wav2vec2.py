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
""" Mindspore Wav2Vec2 model. """

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal, Uniform

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ....utils import (
    cached_file,
    logging,
)

from .configuration_wav2vec2 import Wav2Vec2Config

__all__ = [
    'WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST',
    'Wav2Vec2PreTrainedModel',
    'Wav2Vec2Model',
    'Wav2Vec2ForPreTraining',
    'Wav2Vec2ForMaskedLM',
    'Wav2Vec2ForCTC',
    'Wav2Vec2ForSequenceClassification',
    'Wav2Vec2ForAudioFrameClassification',
    'Wav2Vec2ForXVector',
]

logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 2

WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"
WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://hf-mirror.com/models?filter=wav2vec2
]


@dataclass
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `Tensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`Tensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`Tensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`):
            Tuple of `Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(Tensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`):
            Tuple of `Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `Tensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `Tensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """
    loss: Optional[Tensor] = None
    projected_states: Tensor = None
    projected_quantized_states: Tensor = None
    codevector_perplexity: Tensor = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    contrastive_loss: Optional[Tensor] = None
    diversity_loss: Optional[Tensor] = None


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


def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    if isinstance(mask_time_indices, Tensor):
        mask_time_indices = mask_time_indices.asnumpy()
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


class Wav2Vec2NoLayerNormConvLayer(nn.Module):

    """
    Wav2Vec2NoLayerNormConvLayer is a Python class representing a convolutional layer without layer normalization for
    the Wav2Vec2 model. This class inherits from nn.Module and is used for processing audio features.

    Attributes:
        config (Wav2Vec2Config): The configuration object for the Wav2Vec2 model.
        layer_id (int): The index of the convolutional layer.
        in_conv_dim (int): The input dimension of the convolutional layer.
        out_conv_dim (int): The output dimension of the convolutional layer.
        conv (nn.Conv1d): The 1D convolutional operation applied to the input.
        activation (function): The activation function used to process the convolutional output.

    Methods:
        __init__: Initializes the Wav2Vec2NoLayerNormConvLayer with the provided configuration and layer index.
        forward: Applies the convolutional and activation operations to the input hidden_states.

    Note:
        This class is part of the Wav2Vec2 model and is specifically designed for processing audio features without
        layer normalization.
    """
    def __init__(self, config: Wav2Vec2Config, layer_id=0):
        """
        __init__(self, config: Wav2Vec2Config, layer_id=0)

        Initializes a new instance of the Wav2Vec2NoLayerNormConvLayer class.

        Args:
            self: The instance of the class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the Wav2Vec2 model.
            layer_id (int, optional): The index of the layer. Defaults to 0. Specifies the layer for which the
                convolutional layer is initialized.

        Returns:
            None.

        Raises:
            ValueError: If the layer_id is less than 0.
            AttributeError: If the layer_id exceeds the maximum index available in the configuration parameters.
            TypeError: If the provided config parameter is not an instance of the Wav2Vec2Config class.
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        """
        Constructs the hidden states using convolutional layer and activation function.

        Args:
            self (Wav2Vec2NoLayerNormConvLayer): The instance of the Wav2Vec2NoLayerNormConvLayer class.
            hidden_states (torch.Tensor): The input hidden states tensor.

        Returns:
            torch.Tensor: The forwarded hidden states after applying convolution and activation.

        Raises:
            TypeError: If the input hidden_states is not a torch.Tensor.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):

    """
    This class represents a convolutional layer with layer normalization in the Wav2Vec2 model.
    It inherits from the nn.Module class.

    Attributes:
        config (Wav2Vec2Config): The configuration object for the Wav2Vec2 model.
        layer_id (int): The ID of the current layer.

    Methods:
        __init__:
            Initializes the Wav2Vec2LayerNormConvLayer with the given configuration and layer ID.

        forward:
            Applies the convolutional layer with layer normalization to the input hidden states.

    """
    def __init__(self, config: Wav2Vec2Config, layer_id=0):
        """
        Initialize the Wav2Vec2LayerNormConvLayer.

        Args:
            config (Wav2Vec2Config): The configuration object containing the parameters for the layer.
            layer_id (int, optional): The ID of the layer. Defaults to 0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        """
        Construct the hidden states using the Wav2Vec2LayerNormConvLayer method.

        Args:
            self (Wav2Vec2LayerNormConvLayer): An instance of the Wav2Vec2LayerNormConvLayer class.
            hidden_states (Tensor): The input hidden states to be processed.
                It should have the shape (batch_size, sequence_length, feature_dim).

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.swapaxes(-2, -1)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):

    """
    This class represents a group normalization convolutional layer used in the Wav2Vec2 model.
    It applies a 1D convolution operation followed by group normalization, activation, and layer normalization to the
    input hidden states.

    Args:
        config (Wav2Vec2Config): The configuration object containing the settings for the Wav2Vec2 model.
        layer_id (int, optional): The index of the convolutional layer in the model. Defaults to 0.

    Attributes:
        in_conv_dim (int): The input dimension of the convolutional layer.
        out_conv_dim (int): The output dimension of the convolutional layer.
        conv (nn.Conv1d): The 1D convolutional layer used to process the hidden states.
        activation (function): The activation function applied to the processed hidden states.
        layer_norm (nn.GroupNorm): The group normalization layer applied to the hidden states.

    Methods:
        forward: Applies the convolutional layer, normalization, activation, and returns the processed hidden states.

    """
    def __init__(self, config: Wav2Vec2Config, layer_id=0):
        """
        Initializes an instance of the Wav2Vec2GroupNormConvLayer class.

        Args:
            self: The current instance of the class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing configuration settings.
            layer_id (int): The index of the convolutional layer within the configuration. Defaults to 0.

        Returns:
            None.

        Raises:
            ValueError: If the layer_id is less than 0.
            KeyError: If the specified activation function in config is not found in the ACT2FN dictionary.
            ValueError: If the specified pad_mode in the nn.Conv1d function is not 'valid'.
        """
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        """
        This method forwards a group normalization convolutional layer for the Wav2Vec2 model.

        Args:
            self (Wav2Vec2GroupNormConvLayer): The instance of the Wav2Vec2GroupNormConvLayer class.
            hidden_states (torch.Tensor): The input tensor representing the hidden states to be processed by the group normalization convolutional layer.

        Returns:
            torch.Tensor: The processed tensor representing the hidden states after passing through the group normalization convolutional layer.

        Raises:
            None.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states.unsqueeze(-1)).squeeze(-1)    # tmfix: GroupNorm only support 4D
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):

    """
    This class represents a positional convolutional embedding layer in the Wav2Vec2 model architecture.
    It inherits from nn.Module and is designed to process hidden states through convolutional and activation operations.

    Attributes:
        config: Wav2Vec2Config
            An instance of Wav2Vec2Config containing configuration parameters for the layer.

    Methods:
        __init__:
            Initializes the Wav2Vec2PositionalConvEmbedding with the provided configuration.

        forward:
            Applies positional convolutional embedding operations on the input hidden_states and returns the
            transformed output.

    Usage:
        Instantiate this class by providing a Wav2Vec2Config object as configuration, then call the forward method
        with hidden states to process them.

    Note:
        This class utilizes a convolutional layer, padding layer, and activation function to process hidden states
        efficiently.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2PositionalConvEmbedding class.

        Args:
            self: An instance of the Wav2Vec2PositionalConvEmbedding class.
            config (Wav2Vec2Config): The configuration object containing various settings for the Wav2Vec2 model.

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
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            bias=True,
        )

        self.conv = F.weight_norm(self.conv, name='weight', dim=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        """
        This method forwards the positional convolutional embedding for the Wav2Vec2 model.

        Args:
            self (Wav2Vec2PositionalConvEmbedding): The instance of the Wav2Vec2PositionalConvEmbedding class.
            hidden_states (array-like): The input hidden states with shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not return any value. The positional convolutional embedding is applied to the
                input hidden states in place.

        Raises:
            ValueError: If the input hidden_states is not in the expected format or shape.
            RuntimeError: If an error occurs during the convolution or activation process.
        """
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):

    """
    This class represents a layer in the Wav2Vec2 model that performs padding removal.

    Wav2Vec2SamePadLayer is a subclass of nn.Module and is designed to remove padding from hidden states in the
    Wav2Vec2 model. It is primarily used in the Wav2Vec2 model for speech recognition tasks.

    Attributes:
        num_pad_remove (int): The number of padding elements to remove from the hidden states.

    Methods:
        __init__: Initializes a new instance of the Wav2Vec2SamePadLayer class.
        forward: Removes padding elements from the hidden states.

    """
    def __init__(self, num_conv_pos_embeddings):
        """
        Initializes an instance of the Wav2Vec2SamePadLayer class.

        Args:
            self (Wav2Vec2SamePadLayer): The current instance of the Wav2Vec2SamePadLayer class.
            num_conv_pos_embeddings (int): The number of convolutional positional embeddings.
                It is used to determine the value of the num_pad_remove attribute.
                The value must be a non-negative integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        """
        Constructs the hidden states of the Wav2Vec2SamePadLayer.

        Args:
            self (Wav2Vec2SamePadLayer): An instance of the Wav2Vec2SamePadLayer class.
            hidden_states (torch.Tensor): The hidden states to be processed.
                Expected shape is (batch_size, sequence_length, hidden_size).
                The hidden states are processed based on the `num_pad_remove` value.

        Returns:
            None.

        Raises:
            None.
        """
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2FeatureEncoder class.

        Args:
            self: The object itself.
            config (Wav2Vec2Config):
                The configuration object for the feature encoder.

                - config.feat_extract_norm (str): The type of normalization to be applied during feature extraction.

                    - 'group': Applies group normalization to the convolutional layers.
                    - 'layer': Applies layer normalization to the convolutional layers.

                - config.num_feat_extract_layers (int): The number of feature extraction layers.

        Returns:
            None.

        Raises:
            ValueError: If `config.feat_extract_norm` is not one of ['group', 'layer'].

        """
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self._requires_grad = True

    def _freeze_parameters(self):
        """
        Freezes the parameters of the Wav2Vec2FeatureEncoder.

        Args:
            self: An instance of the Wav2Vec2FeatureEncoder class.

        Returns:
            None.

        Raises:
            None.
        """
        for _, param in self.parameters_and_names():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        """
        Method 'forward' in the class 'Wav2Vec2FeatureEncoder' forwards the hidden states from the input values
        using convolutional layers.

        Args:
            self (object): The instance of the class.
            input_values (tensor): The input values for forwarding hidden states. It is expected to be a 2D tensor.

        Returns:
            tensor: The forwarded hidden states after passing through the convolutional layers.

        Raises:
            None
        """
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class Wav2Vec2FeatureExtractor(Wav2Vec2FeatureEncoder):

    """
    Wav2Vec2FeatureExtractor is a class that represents a feature extractor for Wav2Vec2 models.
    It is designed to extract features from audio data for use in Wav2Vec2 models.

    This class inherits from Wav2Vec2FeatureEncoder, and it is recommended to use Wav2Vec2FeatureEncoder instead of
    this class, as Wav2Vec2FeatureExtractor has been deprecated.

    Please refer to the documentation for Wav2Vec2FeatureEncoder for feature extraction and encoding in Wav2Vec2 models.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        This method initializes an instance of the Wav2Vec2FeatureExtractor class.

        Args:
            self: The instance of the class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the feature extractor.

        Returns:
            None.

        Raises:
            FutureWarning: If the class Wav2Vec2FeatureExtractor is used, a FutureWarning is raised indicating that
                the class has been depreciated. It is recommended to use the base
                class instead.
        """
        super().__init__(config)
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class Wav2Vec2FeatureProjection(nn.Module):

    """
    Wav2Vec2FeatureProjection is a Python class that represents a feature projection module for Wav2Vec2.
    This class inherits from nn.Module and contains methods for initializing the feature projection and forwarding the
    hidden states.

    The __init__ method initializes the feature projection module by setting up layer normalization, dense projection,
    and dropout.

    The forward method applies layer normalization to the hidden states, projects the normalized states using dense
    projection, and applies dropout to the projected states before returning the hidden states and the normalized
    hidden states.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes the Wav2Vec2FeatureProjection class.

        Args:
            self: The instance of the Wav2Vec2FeatureProjection class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the Wav2Vec2 feature projection. It specifies the configuration for the layer
                normalization, projection, and dropout layers.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type Wav2Vec2Config.
            ValueError: If the config.conv_dim[-1] is not valid or if the config.hidden_size is not valid.
            RuntimeError: If an error occurs during the initialization of layer normalization, projection,
                or dropout layers.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(p=config.feat_proj_dropout)

    def forward(self, hidden_states):
        """
        This method forwards the hidden states by applying layer normalization, projection, and dropout.

        Args:
            self (Wav2Vec2FeatureProjection): The instance of the Wav2Vec2FeatureProjection class.
            hidden_states (Tensor): The input hidden states to be processed. It should be a tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            Tuple[Tensor, Tensor]:
                A tuple containing two tensors:

                - hidden_states (Tensor): The processed hidden states after applying layer normalization, projection,
                and dropout.
                - norm_hidden_states (Tensor): The normalized hidden states obtained after applying layer normalization.

        Raises:
            None.
        """
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
    ):
        """
        Initializes an instance of the Wav2Vec2Attention class.

        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Whether the attention module is used as a decoder. Defaults to False.
            bias (bool, optional): Whether to include bias in linear projections. Defaults to True.
            is_causal (bool, optional): Whether the attention is causal. Defaults to False.
            config (Optional[Wav2Vec2Config], optional): The configuration object. Defaults to None.

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

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' is defined in the class 'Wav2Vec2Attention' and is used to reshape the input tensor to
        the specified shape.

        Args:
            tensor (Tensor): The input tensor to be reshaped. It should be of type Tensor.
            seq_len (int): The length of the sequence. It should be an integer.
            bsz (int): The batch size. It should be an integer.

        Returns:
            None.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
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
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
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


class Wav2Vec2FeedForward(nn.Module):

    """
    Wav2Vec2FeedForward is a class representing the feedforward network for the Wav2Vec2 model.
    This class inherits from nn.Module and contains methods for initializing the network and forwarding the
    feedforward layers.

    The __init__ method initializes the feedforward network with the provided configuration.
    It sets up the intermediate dropout, intermediate dense, intermediate activation function, output dense, and output
    dropout layers based on the configuration parameters.

    The forward method takes hidden states as input and processes them through the intermediate dense layer,
    intermediate activation function, intermediate dropout layer, output dense layer, and output dropout layer.
    It then returns the processed hidden states.

    Note:
        This docstring is based on the provided code snippet and may need to be updated with additional information once
        the entire class implementation is available.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initialize the Wav2Vec2FeedForward class.

        Args:
            self: Instance of the class.
            config (Wav2Vec2Config): Configuration object containing parameters for initialization.
                The config parameter is of type Wav2Vec2Config and holds the configuration settings required for
                initializing the feed-forward module.
                It is expected to contain the following attributes:

                - activation_dropout (float): Dropout probability for intermediate layers.
                - hidden_size (int): Size of the hidden layers.
                - intermediate_size (int): Size of the intermediate layer.
                - hidden_act (str or function): Activation function for the hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.intermediate_dropout = nn.Dropout(p=config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(self, hidden_states):
        """
        Constructs the feed-forward network for the Wav2Vec2 model.

        Args:
            self (Wav2Vec2FeedForward): An instance of the Wav2Vec2FeedForward class.
            hidden_states (torch.Tensor): The input hidden states to be passed through the feed-forward network.

        Returns:
            torch.Tensor: The output hidden states after passing through the feed-forward network.

        Raises:
            TypeError: If the input hidden_states is not of type torch.Tensor.
            ValueError: If the input hidden_states does not have a rank of 2.

        This method takes the input hidden states and passes them through a feed-forward network consisting of several
        layers. The feed-forward network is forwarded using intermediate dense layers, activation functions,
        and dropout layers. The hidden_states are first passed through the intermediate dense layer, followed by the
        intermediate activation function and dropout layer. The resulting hidden_states are then passed through the
        output dense layer and another dropout layer. The final output hidden_states are returned.
        Note that the input hidden_states must be a tensor of rank 2, representing a batch of hidden states.
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2EncoderLayer(nn.Module):

    """A class representing an encoder layer of the Wav2Vec2 model.

    The Wav2Vec2EncoderLayer class inherits from the nn.Module class and implements the functionality of a single encoder
    layer in the Wav2Vec2 model architecture. It consists of multiple sub-modules, including an attention mechanism,
    dropout layers, layer normalization, and a feed-forward neural network.

    Attributes:
        attention (Wav2Vec2Attention): The attention mechanism used in the layer.
        dropout (nn.Dropout): The dropout layer applied to the hidden states.
        layer_norm (nn.LayerNorm): The layer normalization applied to the hidden states.
        feed_forward (Wav2Vec2FeedForward): The feed-forward neural network used in the layer.
        final_layer_norm (nn.LayerNorm): The final layer normalization applied to the hidden states.

    Methods:
        forward(hidden_states, attention_mask=None, output_attentions=False):
            Applies the forward pass of the encoder layer.

            Args:

            - hidden_states (Tensor): The input hidden states.
            - attention_mask (Tensor, optional): The attention mask to apply to the attention mechanism (default: None).
            - output_attentions (bool, optional): Whether to return the attention weights (default: False).

            Returns:

            - outputs (tuple): A tuple containing the output hidden states. If output_attentions is True, the tuple
            also contains the attention weights.

    Note:
        The Wav2Vec2EncoderLayer class is designed to be used within the Wav2Vec2Encoder class, which stacks multiple
        encoder layers to form the complete Wav2Vec2 model.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a Wav2Vec2EncoderLayer instance.

        Args:
            self (Wav2Vec2EncoderLayer): The instance of the Wav2Vec2EncoderLayer class.
            config (Wav2Vec2Config):
                An instance of Wav2Vec2Config containing configuration parameters for the encoder layer.

                - Wav2Vec2Config.hidden_size (int): The hidden size for the encoder layer.
                - Wav2Vec2Config.num_attention_heads (int): The number of attention heads in the attention mechanism.
                - Wav2Vec2Config.attention_dropout (float): The dropout probability for the attention mechanism.
                - Wav2Vec2Config.hidden_dropout (float): The dropout probability for the hidden layers.
                - Wav2Vec2Config.layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        """
        Constructs the Wav2Vec2EncoderLayer.

        This method applies the Wav2Vec2EncoderLayer to the input hidden_states. It performs attention, residual
        connections, layer normalization, feed-forward, and final layer normalization.

        Args:
            self (Wav2Vec2EncoderLayer): The instance of the Wav2Vec2EncoderLayer class.
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): The attention mask of shape (batch_size, sequence_length).
                Defaults to None.
            output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.

        Returns:
            tuple: A tuple containing the hidden states of shape (batch_size, sequence_length, hidden_size).
                If output_attentions is True, the tuple also contains the attention weights of shape (batch_size,
                num_heads, sequence_length, sequence_length).

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


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):

    """
    This class represents an encoder layer in the Wav2Vec2 model with stable layer normalization.
    It inherits from the nn.Module class.

    Attributes:
        attention (Wav2Vec2Attention): An instance of the Wav2Vec2Attention class for attention mechanism.
        dropout (nn.Dropout): An instance of the nn.Dropout class for dropout regularization.
        layer_norm (nn.LayerNorm): An instance of the nn.LayerNorm class for stable layer normalization.
        feed_forward (Wav2Vec2FeedForward): An instance of the Wav2Vec2FeedForward class for feed-forward layer.
        final_layer_norm (nn.LayerNorm): An instance of the nn.LayerNorm class for stable layer normalization of final
            output.
        adapter_layer (Wav2Vec2AttnAdapterLayer or None): An instance of the Wav2Vec2AttnAdapterLayer class for adapter
            layer, if provided. None otherwise.

    Methods:
        forward:
            Applies the encoder layer operations on the input hidden states.

            Args:

            - hidden_states (Tensor): The input hidden states.
            - attention_mask (Optional[Tensor]): The attention mask tensor, if provided. Defaults to None.
            - output_attentions (bool): Whether to output attention weights. Defaults to False.

            Returns:

            - Tuple[Tensor, Union[Tensor, None]]: A tuple containing the final hidden states and optionally the
            attention weights, if output_attentions is True.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2EncoderLayerStableLayerNorm class.

        Args:
            self: The instance of the class.
            config (Wav2Vec2Config): The configuration object containing the settings for the encoder layer.
                It should be an instance of the Wav2Vec2Config class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Constructs the Wav2Vec2EncoderLayerStableLayerNorm.

        Args:
            self: Instance of the Wav2Vec2EncoderLayerStableLayerNorm class.
            hidden_states (Tensor): The input hidden states to be processed by the encoder layer.
            attention_mask (Optional[Tensor]): Optional tensor representing the attention mask.
                Defaults to None. If provided, masks certain elements in the attention computation.
            output_attentions (bool): Flag indicating whether to output attention weights during computation.
                Defaults to False.

        Returns:
            Tuple: A tuple containing the processed hidden states and optionally the attention weights.

        Raises:
            None.
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


class Wav2Vec2Encoder(nn.Module):

    """
    A class representing the Wav2Vec2Encoder in the Wav2Vec2 model architecture.

    The Wav2Vec2Encoder is responsible for encoding the input hidden states with positional embeddings and applying
    a series of Wav2Vec2EncoderLayer for feature extraction.

    Attributes:
        config (Wav2Vec2Config): The configuration for the Wav2Vec2 model.
        pos_conv_embed (Wav2Vec2PositionalConvEmbedding): The positional convolutional embedding layer.
        layer_norm (nn.LayerNorm): The layer normalization layer.
        dropout (nn.Dropout): The dropout layer.
        layers (nn.ModuleList): The list of Wav2Vec2EncoderLayer instances.

    Methods:
        forward(hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
            Applies the Wav2Vec2Encoder layer-wise to the hidden states.

            Args:

            - hidden_states (Tensor): The input hidden states.
            - attention_mask (Optional[Tensor], optional): The attention mask tensor. Defaults to None.
            - output_attentions (bool, optional): Whether to output the attentions. Defaults to False.
            - output_hidden_states (bool, optional): Whether to output the hidden states. Defaults to False.
            - return_dict (bool, optional): Whether to return a BaseModelOutput dictionary. Defaults to True.

            Returns:

            - BaseModelOutput or Tuple[Tensor, Tuple[Tensor], Tuple[Tensor]]: The encoded hidden states, all hidden
            states (if output_hidden_states=True), and all self-attentions (if output_attentions=True).
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes the Wav2Vec2Encoder class.

        Args:
            self: The instance of the class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the encoder. It specifies the configuration for the Wav2Vec2 model, such as hidden size,
                layer normalization epsilon, hidden dropout probability, and the number of hidden layers.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions explicitly. However, exceptions may be raised during the
                initialization of the Wav2Vec2PositionalConvEmbedding, nn.LayerNorm, nn.Dropout, and nn.ModuleList objects.
        """
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Constructs the Wav2Vec2Encoder.

        Args:
            self (Wav2Vec2Encoder): The instance of the Wav2Vec2Encoder class.
            hidden_states (Tensor): The input hidden states. A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[Tensor]): An optional tensor specifying the attention mask. Defaults to None.
            output_attentions (bool): Whether to output attentions. Defaults to False.
            output_hidden_states (bool): Whether to output hidden states. Defaults to False.
            return_dict (bool): Whether to return a dictionary. Defaults to True.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_states tensor has invalid shape or type.
            ValueError: If the attention_mask tensor has invalid shape or type.
            TypeError: If the output_attentions or output_hidden_states parameters are not of type bool.
            TypeError: If the return_dict parameter is not of type bool.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * float(ops.finfo(hidden_states.dtype).min)
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


class Wav2Vec2EncoderStableLayerNorm(nn.Module):

    """
    Wav2Vec2EncoderStableLayerNorm is a Python class that represents an encoder with stable layer normalization for
    the Wav2Vec2 model. This class inherits from the nn.Module module.

    This class initializes with a Wav2Vec2Config object and forwards a series of encoder layers with stable
    layer normalization. The encoder layers operate on the input hidden states and optionally apply
    attention masks, producing hidden states with added positional embeddings and layer normalization.

    The forward method applies the encoder layers to the input hidden states, handling attention masks,
    outputting hidden states, and attentions based on the specified configurations.

    This class provides functionalities for building and using a stable layer normalization encoder for the Wav2Vec2
    model, supporting various output options and configurations.

    For detailed information on the class methods and usage, please refer to the specific method docstrings within
    the source code.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes an instance of the Wav2Vec2EncoderStableLayerNorm class.

        Args:
            self: The object instance.
            config (Wav2Vec2Config): The configuration object for the Wav2Vec2 model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        Constructs the Wav2Vec2EncoderStableLayerNorm.

        Args:

        - hidden_states: The input hidden states of shape (batch_size, sequence_length, hidden_size).
        - attention_mask: Optional attention mask of shape (batch_size, sequence_length).
        It is used to mask the attention scores.
        - output_attentions: Boolean flag indicating whether to output attention weights. Defaults to False.
        - output_hidden_states: Boolean flag indicating whether to output hidden states of all layers. Defaults to False.
        - return_dict: Boolean flag indicating whether to return a dictionary as output. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * float(ops.finfo(hidden_states.dtype).min)
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
            dropout_probability = ops.rand([])

            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
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


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2GumbelVectorQuantizer class.

        Args:
            self: The instance of the Wav2Vec2GumbelVectorQuantizer class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing configuration parameters
                for the vector quantizer.

                - num_codevector_groups (int): The number of codevector groups.
                - num_codevectors_per_group (int): The number of codevectors per group.
                - codevector_dim (int): The dimension of the codevectors.

        Returns:
            None.

        Raises:
            ValueError: If `config.codevector_dim` is not divisible by `config.num_codevector_groups` for concatenation.
        """
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = Parameter(
            ops.zeros((1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups))
        )
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        """
        Compute the perplexity of given probability distribution.

        Args:
            probs (Tensor): The input probability distribution. It should be a tensor of shape (N, D) where N is the
                number of elements and D is the dimensionality of the distribution. mask (Tensor, optional):
                A boolean tensor of the same shape as probs, indicating which elements to include in the computation.
                If provided, only the elements where mask is True will be considered. Defaults to None.

        Returns:
            None: This method does not return anything but updates the internal state of the class.

        Raises:
            ValueError: If the shape of probs and mask do not match.
            ValueError: If the dimensionality of probs is not 2.
        """
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = ops.where(mask_extended, probs, ops.zeros_like(probs))
            marginal_probs = probs.sum(axis=0) / mask.sum()
        else:
            marginal_probs = probs.mean(axis=0)

        perplexity = ops.exp(-ops.sum(marginal_probs * ops.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        '''
        Constructs codevectors and computes perplexity for Wav2Vec2GumbelVectorQuantizer.

        Args:
            self: The instance of the Wav2Vec2GumbelVectorQuantizer class.
            hidden_states (tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).
            mask_time_indices (tensor, optional): A binary mask tensor of shape (batch_size, sequence_length) where
                1s indicate valid time indices and 0s indicate masked time indices. Default is None.

        Returns:
            tuple:
                A tuple containing:

                - codevectors (tensor): The forwarded codevectors with shape (batch_size, sequence_length, -1).
                - perplexity (tensor): The computed perplexity.

        Raises:
            ValueError: If the input hidden_states tensor has an invalid shape.
            RuntimeError: If the function encounters a runtime error during computation.
        '''
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = ops.gumbel_softmax(
                hidden_states.float(), tau=float(self.temperature), hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = ops.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            # NOTE: 把 hidden_states 变成 hardsoftmax(dim=-1) 形式
            codevector_idx = ops.argmax(hidden_states, dim=-1)      # (364) => (364, 1)
            x = hidden_states.new_zeros(hidden_states.shape)    # (364, 320)
            index = codevector_idx.view(-1, 1)
            update = ops.ones_like(index, dtype=hidden_states.dtype)    # fill with onehot
            codevector_probs = ops.scatter(x, -1, index, update)
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1) # (182, 2, 320)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity


class Wav2Vec2Adapter(nn.Module):

    """
    Wav2Vec2Adapter is a class that represents an adapter layer for adapting the hidden states of a Wav2Vec2 model.
    This class inherits from nn.Module and implements methods for initializing and forwarding the adapter layer.

    Attributes:
        proj (nn.Linear or None): A dense layer used for projecting hidden states if output_hidden_size is
            different from hidden_size.
        proj_layer_norm (nn.LayerNorm or None): A layer normalization module applied after projection if needed.
        layers (nn.ModuleList): A list of Wav2Vec2AdapterLayer instances representing adapter layers.
        layerdrop (float): The probability of dropping a layer during training.

    Methods:
        __init__: Initializes the Wav2Vec2Adapter object with the provided configuration.
        forward: Applies the adapter layer transformations to the input hidden states.

    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2Adapter class.

        Args:
            self: The current instance of the class.
            config (Wav2Vec2Config): An instance of Wav2Vec2Config containing configuration parameters for the adapter.
                This parameter is required for initializing the adapter and must be an instance of Wav2Vec2Config.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type Wav2Vec2Config.
            ValueError: If the output_hidden_size in the config parameter does not match the hidden_size.
        """
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList([Wav2Vec2AdapterLayer(config) for _ in range(config.num_adapter_layers)])
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        """
        This method forwards the hidden states by applying transformations and layers.

        Args:
            self (object): The instance of the Wav2Vec2Adapter class.
            hidden_states (numpy.ndarray): The input hidden states to be processed.
                It is expected to be a 3D array with shape (batch_size, sequence_length, hidden_size).

        Returns:
            numpy.ndarray: The processed hidden states with shape (batch_size, sequence_length, hidden_size).

        Raises:
            None
        """
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


class Wav2Vec2AdapterLayer(nn.Module):

    '''
    Wav2Vec2AdapterLayer is a Python class that represents an adapter layer for the Wav2Vec2 model.
    This class inherits from nn.Module.

    The adapter layer contains methods for initialization and forwardion.

    The __init__ method initializes the adapter layer with the provided configuration. It sets up a 1D convolutional
    layer with specified parameters such as kernel size, stride, padding, and bias.

    The forward method takes hidden_states as input and applies the convolutional layer followed by the
    gated linear unit (GLU) activation function. It then returns the processed hidden states.

    This class provides functionality for creating and processing adapter layers within the Wav2Vec2 model.
    '''
    def __init__(self, config: Wav2Vec2Config):
        """
        __init__

        Initializes a new instance of the Wav2Vec2AdapterLayer class.

        Args:
            self: The instance of the Wav2Vec2AdapterLayer class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the adapter layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
            bias=True,
        )

    def forward(self, hidden_states):
        """
        Method to forward the Wav2Vec2AdapterLayer.

        Args:
            self (Wav2Vec2AdapterLayer): The instance of the Wav2Vec2AdapterLayer class.
            hidden_states (Tensor): The input hidden states to be processed. It should be a tensor.

        Returns:
            Tensor: The processed hidden states after applying convolution and gated linear units (GLU) operation.

        Raises:
            None.
        """
        hidden_states = self.conv(hidden_states)
        hidden_states = F.glu(hidden_states, dim=1)
        return hidden_states


class Wav2Vec2AttnAdapterLayer(nn.Module):

    """
    This class represents a single layer of an attention adapter module in the Wav2Vec2 model. The adapter module is
    designed to enhance the training throughput by directly implementing the adapter modules with 3D tensor weights as
    parameters, without using ModuleList.

    Attributes:
        input_dim (int): The dimension of the input tensor to the adapter module.
        hidden_dim (int): The hidden dimension of the adapter module.
        norm (nn.LayerNorm): A layer normalization module to normalize the hidden states.
        linear_1 (nn.Linear): A linear transformation module that maps the hidden states to the input dimension.
        act_fn (nn.ReLU): An activation function module that applies the ReLU activation to the hidden states.
        linear_2 (nn.Linear): A linear transformation module that maps the hidden states back to the hidden dimension.

    Methods:
        forward:
            Applies the attention adapter layer operations to the input hidden states tensor.

            Args:

            -  hidden_states (Tensor): The input hidden states tensor.
            Returns:

            - Tensor: The output hidden states tensor after applying the attention adapter layer operations.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        super().__init__()
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        self.act_fn = nn.ReLU()
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: Tensor):
        """
        Method: forward

        Description:
        Constructs the adaptation layer for the Wav2Vec2AttnAdapterModel.

        Args:
            self: (Wav2Vec2AttnAdapterLayer) The instance of the Wav2Vec2AttnAdapterLayer class.
            hidden_states: (Tensor) The input hidden states to be processed by the adaptation layer.

        Returns:
            None

        Raises:
            ValueError: If the input hidden_states tensor is empty or invalid.
            TypeError: If the input hidden_states is not of type Tensor.
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
        if isinstance(cell, Wav2Vec2ForPreTraining):
            cell.project_hid._is_initialized = True
            cell.project_q._is_initialized = True
        # gumbel softmax requires special init
        elif isinstance(cell, Wav2Vec2GumbelVectorQuantizer):
            cell.weight_proj.weight.set_data(initializer(Normal(1.0), cell.weight_proj.weight.shape, cell.weight_proj.weight.dtype))
            cell.weight_proj.bias.set_data(initializer('zeros', cell.weight_proj.bias.shape, cell.weight_proj.bias.dtype))
            cell.codevectors.set_data(initializer('uniform', cell.codevectors.shape, cell.codevectors.dtype))
        elif isinstance(cell, Wav2Vec2PositionalConvEmbedding):
            cell.conv.weight.set_data(
                initializer(Normal(2 * math.sqrt(1 / (cell.conv.kernel_size[0] * cell.conv.in_channels))),
                            cell.conv.weight.shape, cell.conv.weight.dtype))
            cell.conv.bias.set_data(initializer('zeros', cell.conv.bias.shape, cell.conv.bias.dtype))
        elif isinstance(cell, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / cell.projection.in_channels)
            cell.projection.weight.set_data(
                initializer(Uniform(k), cell.projection.weight.shape, cell.projection.weight.dtype))
            cell.projection.bias.set_data(
                initializer(Uniform(k), cell.projection.bias.shape, cell.projection.bias.dtype))
        elif isinstance(cell, nn.Linear):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Conv1d):
            cell.weight.set_data(initializer('he_normal', cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(Uniform(k), cell.bias.shape, cell.bias.dtype))

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[Tensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pyops.org/docs/stable/generated/ops.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: Tensor, add_adapter=None
    ):
        """
        This method calculates the attention mask for the feature vectors in a Wav2Vec2 model.

        Args:
            self (Wav2Vec2PreTrainedModel): The instance of the Wav2Vec2PreTrainedModel class.
            feature_vector_length (int): The length of the feature vectors.
            attention_mask (Tensor): The attention mask tensor.
            add_adapter (Optional): An optional parameter to add adapter.

        Returns:
            attention_mask (Tensor): The attention mask tensor for the feature vectors.

        Raises:
            None.
        """
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

    def _get_adapters(self):
        """
        Method _get_adapters in the class Wav2Vec2PreTrainedModel.

        Args:
            self (object): The instance of the class Wav2Vec2PreTrainedModel.

        Returns:
            dict: A dictionary containing adapter weights.
                The keys are composed of the parameter names from the adapter layers and the LM head, and the values are
                the corresponding parameters.

        Raises:
            ValueError: If the 'adapter_attn_dim' attribute in 'config' is not defined, a ValueError is raised with
                a message indicating that the class has no adapter layers and prompting to define
                'config.adapter_attn_dim'.
        """
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.")

        adapter_weights = {}
        for name, module in self.parameters_and_names():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.parameters_and_names():
                    adapter_weights[".".join([name, param_name])] = param

        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.parameters_and_names():
                adapter_weights[".".join(["lm_head", name])] = param

        return adapter_weights

    def init_adapter_layers(self):
        """
        (Re-)initialize attention adapter layers and lm head for adapter-only fine-tuning
        """
        # init attention adapters
        for module in self.cells():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)

        # init lm head
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)

    def load_adapter(self, target_lang: str, force_load=True, **kwargs):
        r"""
        Load a language adapter model from a pre-trained adapter model.

        Parameters:
            target_lang (`str`):
                Has to be a language id of an existing adapter weight. Adapter weights are stored in the format
                adapter.<lang>.safetensors or adapter.<lang>.bin
            force_load (`bool`, defaults to `True`):
                Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on hf-mirror.com, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

        Activate the special ["offline-mode"](https://hf-mirror.com/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Example:
            ```python
            >>> from transformers import Wav2Vec2ForCTC, AutoProcessor
            ...
            >>> ckpt = "facebook/mms-1b-all"
            >>> processor = AutoProcessor.from_pretrained(ckpt)
            >>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
            >>> # set specific language
            >>> processor.tokenizer.set_target_lang("spa")
            >>> model.load_adapter("spa")
            ```
        """
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"Cannot load_adapter for {target_lang} if `config.adapter_attn_dim` is not defined.")

        if target_lang == self.target_lang and not force_load: # pylint: disable=access-member-before-definition
            logger.warning(f"Adapter weights are already set to {target_lang}.")
            return

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        use_safetensors = kwargs.pop("use_safetensors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        model_path_or_id = self.config._name_or_path
        state_dict = None

        # 1. Let's first try loading a safetensors adapter weight
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    cache_dir=cache_dir,
                )

                # state_dict = safe_load_file(weight_path)
                state_dict = None
            except EnvironmentError:
                if use_safetensors:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise

            except Exception as exc:
                # For any other exception, we throw a generic error.
                if use_safetensors:
                    raise EnvironmentError(
                        f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                        " from 'https://hf-mirror.com/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                        f" directory containing a file named {filepath}."
                    ) from exc

        # 2. If this didn't work let's try loading a PyTorch adapter weight
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    cache_dir=cache_dir,
                )

                weights_only_kwarg = {"weights_only": True}
                state_dict = ops.load(
                    weight_path,
                    map_location="cpu",
                    **weights_only_kwarg,
                )

            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise

            except Exception as exc:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                    " from 'https://hf-mirror.com/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                    f" directory containing a file named {filepath}."
                ) from exc

        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())

        if len(unexpected_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.")
        elif len(missing_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.")

        # make sure now vocab size is correct
        target_vocab_size = state_dict["lm_head.weight"].shape[0]
        if target_vocab_size != self.config.vocab_size:
            self.lm_head = nn.Linear(
                self.config.output_hidden_size, target_vocab_size, dtype=self.dtype
            )
            self.config.vocab_size = target_vocab_size

        # make sure that adapter weights are put in exactly the same precision and device placement and overwritten adapter weights
        state_dict = {k: v.to(adapter_weights[k]) for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

        # set target language corectly
        self.target_lang = target_lang


class Wav2Vec2Model(Wav2Vec2PreTrainedModel):

    """
    The `Wav2Vec2Model` class is a Python class that represents a Wav2Vec2 model for speech recognition.
    It is a subclass of the `Wav2Vec2PreTrainedModel` class.

    Wav2Vec2Model inherits the following attributes and methods from the parent class:

    - `config`: An instance of the `Wav2Vec2Config` class, containing the configuration parameters for the model.
    - `feature_extractor`: An instance of the `Wav2Vec2FeatureEncoder` class, responsible for extracting features
    from the input waveform.
    - `feature_projection`: An instance of the `Wav2Vec2FeatureProjection` class, responsible for projecting the
    extracted features.
    - `encoder`: An instance of the `Wav2Vec2Encoder` or `Wav2Vec2EncoderStableLayerNorm` class, responsible for
    encoding the hidden states.
    - `adapter`: An instance of the `Wav2Vec2Adapter` class, used to adapt the hidden states (optional).
    - `post_init()`: A method called after the initialization of the model.

    The `Wav2Vec2Model` class also defines the following methods:

    - `freeze_feature_extractor`: Disables the gradient computation for the feature encoder, preventing its parameters
    from being updated during training.
    - `freeze_feature_encoder`: Disables the gradient computation for the feature encoder, preventing its parameters
    from being updated during training.
    - `_mask_hidden_states`: Masks extracted features along
    the time axis and/or the feature axis according to SpecAugment.
    - `forward`: Constructs the model by processing the input values and returns the model outputs.

    Please note that the `freeze_feature_extractor()` method is deprecated.
    The equivalent `freeze_feature_encoder()` method should be used instead.

    For more information about the Wav2Vec2 model, please refer to the official paper [SpecAugment]
    (https://arxiv.org/abs/1904.08779).
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2Model class.

        Args:
            self: The instance of the Wav2Vec2Model class.
            config (Wav2Vec2Config): An instance of the Wav2Vec2Config class containing the configuration parameters
                for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type Wav2Vec2Config.
            ValueError: If the config parameters mask_time_prob or mask_feature_prob are less than 0.0.
            ValueError: If the config parameter do_stable_layer_norm is not a boolean value.
            ValueError: If the config parameter hidden_size is not defined.
            ValueError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = Parameter(initializer(Uniform(), (config.hidden_size,), dtype=mindspore.float32))

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

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

    def forward(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        mask_time_indices: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        """
        Constructs the Wav2Vec2 model for processing input audio data.

        Args:
            self (Wav2Vec2Model): The instance of the Wav2Vec2Model class.
            input_values (Optional[Tensor]): The input audio data values with shape (batch_size, audio_length).
            attention_mask (Optional[Tensor]): The attention mask for the input audio data with shape
                (batch_size, audio_length).
            mask_time_indices (Optional[Tensor]): The mask for time indices with shape (batch_size, audio_length).
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool]): Whether to return a dictionary of output. Defaults to None.

        Returns:
            Union[Tuple, Wav2Vec2BaseModelOutput]: The forwarded model output, which can be a tuple or a
                Wav2Vec2BaseModelOutput object.

        Raises:
            ValueError: If the input_values and attention_mask have mismatched shapes.
            TypeError: If the input_values or attention_mask is not a Tensor.
            RuntimeError: If the encoder fails to process the input audio data.
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


class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):

    """Wav2Vec2ForPreTraining

    This class represents a pre-training model for Wav2Vec2, which is used for pre-training the Wav2Vec2 model.
    It includes methods for setting Gumbel softmax temperature, freezing the feature encoder, computing contrastive
    logits, and forwarding the model for pre-training.

    Methods:
        set_gumbel_temperature: Set the Gumbel softmax temperature to a given value. Only necessary for training.
        freeze_feature_extractor: Disable gradient computation for the feature encoder to prevent parameter updates
            during training.
        freeze_feature_encoder: Disable gradient computation for the feature encoder to prevent parameter updates
            during training.
        compute_contrastive_logits: Compute logits for contrastive loss based on cosine similarity between features
            and apply temperature.
        forward: Construct the model for pre-training, including masking features for contrastive loss.

    Attributes:
        wav2vec2: Wav2Vec2Model instance for the Wav2Vec2 model.
        dropout_features: Dropout layer for feature vectors.
        quantizer: Wav2Vec2GumbelVectorQuantizer instance for quantization.
        project_hid: Dense layer for projecting hidden states.
        project_q: Dense layer for projecting quantized features.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2ForPreTraining class.

        Args:
            self: The instance of the Wav2Vec2ForPreTraining class.
            config (Wav2Vec2Config): The configuration object for the Wav2Vec2 model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(p=config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: Tensor,
        negative_features: Tensor,
        predicted_features: Tensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = ops.cat([target_features, negative_features], dim=0)
        logits = ops.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(target_features)
        # apply temperature
        logits = logits / temperature
        return logits

    def forward(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        mask_time_indices: Optional[Tensor] = None,
        sampled_negative_indices: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
        r"""
        Args:
            mask_time_indices (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
                masked extracted features in *config.proj_codevector_dim* space.
            sampled_negative_indices (`Tensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
                Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
                Required input for pre-training.

        Returns:
            Union[Tuple, Wav2Vec2ForPreTrainingOutput]

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
            >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
            >>> from datasets import load_dataset
            ...
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            >>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
            ...
            >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1
            ...
            >>> # compute masked indices
            >>> batch_size, raw_sequence_length = input_values.shape
            >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
            >>> mask_time_indices = _compute_mask_indices(
            ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
            ... )
            >>> sampled_negative_indices = _sample_negative_indices(
            ...     features_shape=(batch_size, sequence_length),
            ...     num_negatives=model.config.num_negatives,
            ...     mask_time_indices=mask_time_indices,
            ... )
            >>> mask_time_indices = Tensor(data=mask_time_indices, device=input_values.device, dtype=mindspore.int64)
            >>> sampled_negative_indices = Tensor(
            ...     data=sampled_negative_indices, device=input_values.device, dtype=mindspore.int64
            ... )
            ...
            >>> with ops.no_grad():
            ...     outputs = model(input_values, mask_time_indices=mask_time_indices)
            ...
            >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
            >>> cosine_sim = ops.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
            ...
            >>> # show that cosine similarity is much higher than random
            >>> cosine_sim[mask_time_indices.to(mindspore.bool_)].mean() > 0.5
            tensor(True)
            >>> # for contrastive loss training model should be put into train mode
            >>> model = model.train()
            >>> loss = model(
            ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
            ... ).loss
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(mindspore.bool_)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                # NOTE: avoid loss NaN
                # float("-inf") => finfo(logits.dtype, 'min') := -3.40282e+38
                logits[1:][neg_is_pos] = -3.40282e+35

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.swapaxes(0, 2).reshape(-1, logits.shape[0])
            target = ((1 - mask_time_indices.long()) * -100).swapaxes(0, 1).flatten()

            contrastive_loss = F.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )


class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):

    """
    This class represents a Wav2Vec2 model for Masked Language Modeling (MLM).
    It is deprecated and should be replaced with `Wav2Vec2ForCTC`.

    The `Wav2Vec2ForMaskedLM` class inherits from the `Wav2Vec2PreTrainedModel` class.

    Attributes:
        `wav2vec2`: The underlying Wav2Vec2Model.
        `dropout`: A dropout layer for regularization.
        `lm_head`: A dense layer for language modeling prediction.

    Methods:
        `__init__`: Initializes a new instance of the `Wav2Vec2ForMaskedLM` class.
        `forward`: Constructs the model for masked language modeling.

    Note:
        This class is deprecated and should be replaced with `Wav2Vec2ForCTC`.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes an instance of the 'Wav2Vec2ForMaskedLM' class.

        Args:
            self: The object instance.
            config (Wav2Vec2Config):
                The configuration object containing various hyperparameters for the model.

                - `config` should be an instance of the 'Wav2Vec2Config' class.
                - This parameter is required.

        Returns:
            None

        Raises:
            FutureWarning: Raised if the class `Wav2Vec2ForMaskedLM` is used, as it is deprecated.
                Recommends using `Wav2Vec2ForCTC` instead.
                This warning is raised as a future version may not support the deprecated class.

        Description:
            This method initializes an instance of the 'Wav2Vec2ForMaskedLM' class. It sets up the model architecture
            and initializes the necessary components. The initialization process includes the following steps:

            1. Calls the parent class '__init__' method using 'super()' to initialize the base class.
            2. Raises a 'FutureWarning' to notify users that the class `Wav2Vec2ForMaskedLM` is deprecated and
            recommends using `Wav2Vec2ForCTC` instead.
            3. Initializes the 'wav2vec2' attribute as an instance of 'Wav2Vec2Model' using the provided 'config'.
            4. Initializes the 'dropout' attribute as an instance of 'nn.Dropout' with the dropout probability specified
            in 'config'.
            5. Initializes the 'lm_head' attribute as an instance of 'nn.Linear' with the hidden size and vocabulary
            size specified in 'config'.
            6. Calls the 'post_init' method to perform any additional post-initialization steps.

        Note:
            The 'Wav2Vec2ForMaskedLM' class is deprecated and may not be supported in future versions. It is recommended
            to use the 'Wav2Vec2ForCTC' class instead.
        """
        super().__init__(config)

        warnings.warn(
            "The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.", FutureWarning
        )

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(p=config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """
        Args:
            self (Wav2Vec2ForMaskedLM): The instance of the Wav2Vec2ForMaskedLM class.
            input_values (Tensor): The input tensor representing the input audio features. Its shape is
                (batch_size, sequence_length, feature_dim).
            attention_mask (Optional[Tensor]): Optional tensor representing the attention mask for the input.
                If provided, should have the shape (batch_size, sequence_length).
            output_attentions (Optional[bool]): Optional flag to indicate whether to return attentions in the output.
                Defaults to None.
            output_hidden_states (Optional[bool]): Optional flag to indicate whether to return hidden states
                in the output. Defaults to None.
            return_dict (Optional[bool]): Optional flag to indicate whether to return the output as a dictionary.
                If not provided, it defaults to the value specified in the configuration.
            labels (Optional[Tensor]): Optional tensor representing the labels for the masked language modeling task.
                Its shape is (batch_size, sequence_length).

        Returns:
            Union[Tuple, MaskedLMOutput]:
                The return value can be either a tuple or a MaskedLMOutput object.

                - If return_dict is False, it returns a tuple containing the logits and, optionally, the hidden states
                and attentions.
                - If return_dict is True, it returns a MaskedLMOutput object containing the logits,
                hidden states, and attentions.

        Raises:
            None
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):

    """
    This class represents a Wav2Vec2 model fine-tuned for Connectionist Temporal Classification (CTC) tasks.
    It inherits from the Wav2Vec2PreTrainedModel, providing methods for initializing the model, tying weights,
    freezing the feature extractor, feature encoder, and base model, as well as forwarding the model
    for inference and training.

    The Wav2Vec2ForCTC class encapsulates the Wav2Vec2 model with additional methods for CTC-specific functionality,
    such as handling labels for CTC, computing CTC loss, and processing input values for CTC tasks.

    The class provides methods for fine-tuning the Wav2Vec2 model for CTC tasks, including freezing specific components
    of the model, as well as forwarding the model for CTC inference and training.

    Additionally, the class provides methods for tying weights and freezing specific components of the model to ensure
    compatibility with adapter weights and to control parameter updates during training.

    This class is designed for fine-tuning the Wav2Vec2 model for CTC tasks, providing a comprehensive set of methods
    for customizing the model's behavior and supporting CTC-specific functionality.
    """
    def __init__(self, config: Wav2Vec2Config, target_lang: Optional[str] = None):
        """
        Initializes a new instance of the Wav2Vec2ForCTC class.

        Args:
            self: The object itself.
            config (Wav2Vec2Config): The configuration for the Wav2Vec2Model.
            target_lang (Optional[str], optional): The target language. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the configuration does not define the vocabulary size of the language model head.

        Note:
            The vocabulary size of the language model head must be defined either by instantiating the model
            with `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)` or by explicitly defining the
            `vocab_size` in the model's configuration.

        """
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(p=config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2 so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Wav2Vec2 never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.wav2vec2.parameters_and_names():
            param.requires_grad = False

    def forward(
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

        outputs = self.wav2vec2(
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
            log_probs = F.log_softmax(logits, dim=-1).swapaxes(0, 1)

            loss, log_alpha = F.ctc_loss(
                log_probs,
                labels,     # flattened_targets
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


class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):

    """
    The `Wav2Vec2ForSequenceClassification` class represents a Wav2Vec2 model for sequence classification tasks.
    It inherits from the `Wav2Vec2PreTrainedModel` class. This class provides methods for initializing the model,
    freezing specific components, and computing the sequence classification output. It also  includes methods for
    handling the feature extractor, feature encoder, and base model. The class supports the forwardion of the sequence
    classification output and provides options for setting various parameters such as attention masks, output attentions,
    output hidden states, and labels.

    Deprecated methods such as `freeze_feature_extractor` and `freeze_base_model` are included along with their
    corresponding replacements. The `forward` method computes the sequence classification/regression loss and handles
    the classification output based on the input values, attention masks, and labels. The class allows for fine-tuning
    the model for sequence classification tasks while providing flexibility in handling different components and
    parameters.

    For detailed information about the class and its methods, refer to the individual method docstrings and the base
    class `Wav2Vec2PreTrainedModel` for additional context and functionality.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2ForSequenceClassification class.

        Args:
            self: The object itself.
            config (Wav2Vec2Config): An instance of Wav2Vec2Config containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            ValueError: Raised if the 'add_adapter' attribute is set to True in the config, as sequence classification
                does not support the use of Wav2Vec2 adapters.
        """
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = Parameter(ops.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.wav2vec2.parameters_and_names():
            param.requires_grad = False

    def forward(
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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, dim=1)
            norm_weights = ops.softmax(self.layer_weights, dim=-1)
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
            loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):

    """
    This class represents a Wav2Vec2 model for audio frame classification. It inherits from the Wav2Vec2PreTrainedModel
    and includes methods for initializing the model, freezing the feature encoder and base model, as well as
    forwarding the model for inference and training.

    Attributes:
        wav2vec2 (Wav2Vec2Model): The Wav2Vec2Model used for audio frame classification.
        classifier (nn.Linear): The classification head for the model.
        num_labels (int): The number of labels for classification.
        layer_weights (Parameter, optional): The weights for weighted layer sum if configured.

    Methods:
        __init__:
            Initializes the Wav2Vec2ForAudioFrameClassification model with the provided configuration.

        freeze_feature_encoder:
            Disables the gradient computation for the feature encoder, preventing its parameters from being updated
            during training.

        freeze_base_model:
            Disables the gradient computation for the base model, preventing its parameters from being updated during
            training while allowing the classification head to be updated.

        forward:
            Constructs the model for inference and training, handling input values, attention masks, labels, and other
            optional parameters. Returns TokenClassifierOutput containing loss, logits, hidden states, and attentions.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes a new instance of the Wav2Vec2ForAudioFrameClassification class.

        Args:
            self: The instance of the class.
            config (Wav2Vec2Config): The configuration object for the Wav2Vec2 model.
                It specifies the parameters and settings for the model initialization.
                Must be an instance of Wav2Vec2Config.

        Returns:
            None.

        Raises:
            ValueError: If the 'config' object has the attribute 'add_adapter' set to True,
                which is not supported for audio frame classification with Wav2Vec2.
        """
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = Parameter(ops.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.init_weights()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for _, param in self.wav2vec2.parameters_and_names():
            param.requires_grad = False

    def forward(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, dim=1)
            norm_weights = ops.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.astype(mindspore.int32)
            loss = F.cross_entropy(logits.view(-1, self.num_labels), ops.argmax(labels.view(-1, self.num_labels), dim=1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AMSoftmaxLoss(nn.Module):

    """
    The AMSoftmaxLoss class represents a neural network cell for computing the AM-Softmax loss. This class inherits
    from nn.Module and provides methods for initializing the loss function and forwarding the computation graph.

    Attributes:
        scale (float): The scale parameter for the AM-Softmax loss function.
        margin (float): The margin parameter for the AM-Softmax loss function.
        num_labels (int): The number of unique labels in the dataset.
        weight (Parameter): The weight parameter for the neural network.

    Methods:
        __init__: Initializes the AMSoftmaxLoss instance with input dimension, number of labels, scale, and margin.

        forward: Constructs the computation graph for the AM-Softmax loss function using the given
            hidden states and labels.

    Note:
        The AMSoftmaxLoss class is designed for use in neural network training and optimization tasks.
    """
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        """
        __init__

        Initializes an instance of the AMSoftmaxLoss class.

        Args:
            self (object): The instance of the class.
            input_dim (int): The dimension of the input features.
            num_labels (int): The number of unique labels for classification.
            scale (float, optional): The scale factor for the angular margin. Defaults to 30.0.
            margin (float, optional): The angular margin value. Defaults to 0.4.

        Returns:
            None.

        Raises:
            ValueError: If input_dim or num_labels are not positive integers.
            TypeError: If scale or margin are not of type float.
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = Parameter(ops.randn(input_dim, num_labels), requires_grad=True)

    def forward(self, hidden_states, labels):
        """
        This method forwards an AMSoftmax loss function.

        Args:
            self (object): The instance of the AMSoftmaxLoss class.
            hidden_states (tensor): A tensor representing the hidden states of the model.
            labels (tensor): A tensor containing the ground truth labels for the corresponding hidden states.
                It is expected that the labels are flattened for processing.

        Returns:
            None.

        Raises:
            ValueError: If the dimensions of the weight tensor and hidden_states tensor are not compatible
                for matrix multiplication.
            RuntimeError: If there is an issue with the normalization operation on the weight or hidden_states tensor.
            ValueError: If the labels tensor does not match the expected shape for one-hot encoding.
            RuntimeError: If there is a problem with the cross-entropy calculation.
        """
        labels = labels.flatten()
        weight = self.weight / ops.norm(self.weight, dim=0, keepdim=True)
        hidden_states = hidden_states / ops.norm(hidden_states, dim=1, keepdim=True)
        cos_theta = ops.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = ops.one_hot(labels, self.num_labels)
        logits = self.scale * ops.where(onehot.bool(), psi, cos_theta)
        loss = F.cross_entropy(logits, labels)
        return loss


class TDNNLayer(nn.Module):

    """TDNNLayer represents a time-delay neural network (TDNN) layer for processing sequential data.
    It inherits from nn.Module and is initialized with a Wav2Vec2Config and an optional layer_id.

    Attributes:
        config (Wav2Vec2Config): The configuration for the Wav2Vec2 model.
        layer_id (int): The index of the TDNN layer.

    Methods:
        forward(hidden_states): Applies the TDNN layer operations to the input hidden_states.

    The TDNNLayer class applies a convolutional layer with specified kernel size and dilation to the input data.
    It then applies a ReLU activation function to the output.

    Note:
        This class is part of the Wav2Vec2 model architecture.

    """
    def __init__(self, config: Wav2Vec2Config, layer_id=0):
        """
        Initializes a TDNNLayer object.

        Args:
            self: The instance of the TDNNLayer class.
            config (Wav2Vec2Config): An instance of Wav2Vec2Config that holds configuration parameters for the layer.
            layer_id (int): An integer representing the ID of the layer. Default is 0. Must be within the range of
                available layers in the configuration.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type Wav2Vec2Config.
            ValueError: If the layer_id is outside the valid range of available layers in the configuration.
        """
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        '''
        Constructs the TDNN layer with the input hidden_states.

        Args:
            self (TDNNLayer): The instance of the TDNNLayer class.
            hidden_states (Tensor): The input hidden states to be processed by the TDNN layer.
                It should be a tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            hidden_states (Tensor): The processed hidden states after applying the TDNN layer operations.
                It will be a tensor of shape (batch_size, out_channels, new_length), where out_channels is the number
                of output channels and new_length is the length of the output sequence.

        Raises:
            TypeError: If the input hidden_states is not a tensor.
            ValueError: If the input hidden_states does not have the expected shape or dimensions.
        '''
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = F.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):

    """
    This class represents a Wav2Vec2 model for extracting x-vector embeddings from audio data. It inherits from the
    Wav2Vec2PreTrainedModel class, and provides methods for freezing specific model components and computing x-vector
    embeddings from input audio data.

    The class contains methods for freezing the feature extractor, freezing the feature encoder, and freezing the base
    model to disable gradient computation for specific model components. Additionally, it includes methods for computing
    the output length of the TDNN layers and for forwarding x-vector embeddings from input audio data.

    The forward method takes input audio data and optional parameters such as attention mask and labels, and returns
    x-vector embeddings along with optional loss and hidden states. The method also supports outputting hidden states
    and attentions based on the configuration settings.

    This class is designed to be used for x-vector extraction tasks and provides flexibility for customizing the model's
    behavior and freezing specific components during training.
    """
    def __init__(self, config: Wav2Vec2Config):
        """
        Initializes an instance of the Wav2Vec2ForXVector class.

        Args:
            self: The instance of the Wav2Vec2ForXVector class.
            config (Wav2Vec2Config): An object of type Wav2Vec2Config containing configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = Parameter(ops.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for named, param in self.wav2vec2.parameters_and_names():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[Tensor, int]):
        """
        Computes the output length of the TDNN layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pyops.org/docs/stable/generated/ops.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    def forward(
        self,
        input_values: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        Args:
            labels (`Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = ops.stack(hidden_states, dim=1)
            norm_weights = ops.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(axis=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(axis=1)
            #std_features = hidden_states.std(axis=1)   # NOTE: buggy API
            std_features = ops.std(hidden_states, dim=1, keepdim=True).squeeze(1)
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
        statistic_pooling = ops.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            labels = labels.astype(mindspore.int32)
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
