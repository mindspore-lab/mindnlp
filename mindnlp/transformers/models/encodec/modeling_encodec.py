# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
""" MindSpore EnCodec model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal, Uniform

from mindnlp.utils import ModelOutput, logging
from mindnlp.modules.weight_norm import weight_norm
from mindnlp.modules.functional import embedding
from ...modeling_utils import PreTrainedModel
from .configuration_encodec import EncodecConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "EncodecConfig"


@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`mindspore.Tensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FlaotTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Encodec.
    """
    audio_codes: mindspore.Tensor = None
    audio_values: mindspore.Tensor = None


@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`mindspore.Tensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`mindspore.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
    """
    audio_codes: mindspore.Tensor = None
    audio_scales: mindspore.Tensor = None


@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`mindspore.Tensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Encodec.
    """
    audio_values: mindspore.Tensor = None


class EncodecConv1d(nn.Cell):
    """Conv1d with asymmetric or causal padding and normalization."""
    def __init__(
        self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
    ):
        """Initialize the EncodecConv1d class.
        
        Args:
            self: The instance of the class.
            config: The configuration object containing various settings.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int, optional): The stride value for the convolution operation. Defaults to 1.
            dilation (int, optional): The dilation value for the convolution operation. Defaults to 1.
        
        Returns:
            None
        
        Raises:
            ValueError: If `norm_type` is not one of the allowed values: `"weight_norm"` or `"time_group_norm"`.
            Warning: If both `stride` and `dilation` are greater than 1.
        
        """
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type

        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "EncodecConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, pad_mode='valid', has_bias=True)
        if self.norm_type == "weight_norm":
            setattr(self, 'conv', weight_norm(self.conv))
        elif self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

    @staticmethod
    def _get_extra_padding_for_conv1d(
        hidden_states: mindspore.Tensor, kernel_size: int, stride: int, padding_total: int = 0
    ) -> int:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: mindspore.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.ops.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode != "reflect":
            return ops.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = ops.pad(hidden_states, (0, extra_pad))
        if mode != 'reflect':
            padded = ops.pad(hidden_states, paddings, mode, value)
        else:
            padded = ops.pad(hidden_states, paddings, mode)

        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def construct(self, hidden_states):
        """
        Method 'construct' in the class 'EncodecConv1d'.
        
        Args:
            self (object): Instance of EncodecConv1d class.
            hidden_states (Tensor):
                Input tensor of shape [batch_size, channels, sequence_length] representing hidden states.
        
        Returns:
            None:
                The method does not return any value but updates the hidden_states tensor after applying convolution
                and normalization.
        
        Raises:
            ValueError: If the normalization type is not supported.
            RuntimeError: If the convolution operation fails.
        """
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states, kernel_size, stride, padding_total)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(hidden_states, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )

        hidden_states = self.conv(hidden_states)
        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Cell):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""
    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        """
        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int, optional): The stride of the convolution. Defaults to 1.
        
        Returns:
            None.
        
        Raises:
            ValueError: If self.norm_type is not one of 'weight_norm' or 'time_group_norm'.
            ValueError: If trim_right_ratio is not equal to 1.0 and causal convolutions are not used.
        """
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        self.conv = nn.Conv1dTranspose(in_channels, out_channels, kernel_size, stride, has_bias=True, pad_mode='valid')
        if config.norm_type == "weight_norm":
            self.conv = weight_norm(self.conv)
        elif config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

    def construct(self, hidden_states):
        """
        This method constructs a 1D transposed convolutional layer for the EncodecConvTranspose1d class.
        
        Args:
            self: An instance of the EncodecConvTranspose1d class.
            hidden_states: A tensor representing the input hidden states to be processed by the
                transposed convolution layer.
        
        Returns:
            None: However, the method modifies the hidden_states tensor to apply the transposed convolution operation.
        
        Raises:
            ValueError: If the norm_type attribute is not recognized or supported.
            RuntimeError: If an error occurs during the transposed convolution operation.
            AttributeError: If the required attributes are not found in the instance of the EncodecConvTranspose1d class.
        """
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2

        padding_left = padding_total - padding_right

        # unpad
        end = hidden_states.shape[-1] - padding_right
        hidden_states = hidden_states[..., padding_left:end]
        return hidden_states


class EncodecLSTM(nn.Cell):
    """
    LSTM without worrying about the hidden state, nor the layout of the data. Expects input as convolutional layout.
    """
    def __init__(self, config, dimension):
        """
        Initializes an instance of the EncodecLSTM class.
        
        Args:
            self (EncodecLSTM): The instance of the EncodecLSTM class.
            config (object): The configuration object containing various settings.
            dimension (int): The dimension of the LSTM input and output.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, config.num_lstm_layers)

    def construct(self, hidden_states):
        """
        Constructs the encoded hidden states using the Long Short-Term Memory (LSTM) algorithm.
        
        Args:
            self (EncodecLSTM): An instance of the EncodecLSTM class.
            hidden_states (torch.Tensor): The hidden states to be encoded.
                Should have shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: The encoded hidden states. Has shape (sequence_length, input_size, batch_size).
        
        Raises:
            None.
        
        Note:
            - The 'hidden_states' tensor is expected to have the batch dimension as the first dimension,
            the sequence dimension as the second dimension, and the input size dimension as the third dimension.
            - The 'hidden_states' tensor is permuted twice to match the expected input format for the LSTM.
            - The LSTM is applied on the permuted 'hidden_states' tensor, and its output is added element-wise to
            the original 'hidden_states' tensor.
            - The resulting tensor is permuted again to match the expected output format.
        """
        hidden_states = hidden_states.permute(2, 0, 1)
        hidden_states = self.lstm(hidden_states)[0] + hidden_states
        hidden_states = hidden_states.permute(1, 2, 0)
        return hidden_states


class EncodecResnetBlock(nn.Cell):
    """
    Residual block from SEANet model as used by EnCodec.
    """
    def __init__(self, config: EncodecConfig, dim: int, dilations: List[int]):
        """
        Initialize the EncodecResnetBlock.
        
        Args:
            self (object): The instance of the class.
            config (EncodecConfig): An object containing configuration parameters for the block.
            dim (int): The dimension of the input data.
            dilations (List[int]): A list of dilation factors for each convolutional layer.
        
        Returns:
            None.
        
        Raises:
            ValueError: Raised if the number of kernel sizes does not match the number of dilations provided.
        """
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.CellList(block)

        if config.use_conv_shortcut:
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def construct(self, hidden_states):
        """
        Constructs the EncodecResnetBlock.
        
        This method applies a series of layers to the given hidden_states to construct the EncodecResnetBlock.
        The method returns the combined result of the residual connection and the output of the layers.
        
        Args:
            self (EncodecResnetBlock): An instance of the EncodecResnetBlock class.
            hidden_states (Tensor): The input hidden states to be passed through the block layers.
                Expected shape: (batch_size, hidden_size).
        
        Returns:
            Tensor: The combined result of the residual connection and the output of the block layers.
                Expected shape: (batch_size, hidden_size).
        
        Raises:
            None.
        """
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Cell):
    """SEANet encoder as used by EnCodec."""
    def __init__(self, config: EncodecConfig):
        """
        This method initializes an instance of the EncodecEncoder class.
        
        Args:
            self: The instance of the EncodecEncoder class.
            config (EncodecConfig):
                An instance of the EncodecConfig class containing configuration parameters for the encoder.

                - audio_channels (int): The number of audio channels.
                - num_filters (int): The number of filters to be used in the encoder.
                - kernel_size (int): The size of the kernel for convolutional layers.
                - upsampling_ratios (list): A list of integers representing the upsampling ratios for each layer.
                - num_residual_layers (int): The number of residual layers to be used in the encoder.
                - dilation_growth_rate (int): The growth rate for the dilation in the residual blocks.
                - hidden_size (int): The size of the hidden layer.
                - last_kernel_size (int): The size of the kernel for the final convolutional layer.

        Returns:
            None:
                The method initializes the layers of the encoder and assigns them to the 'layers' attribute of the
                EncodecEncoder instance.

        Raises:
            None.
        """
        super().__init__()
        model = [EncodecConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            # Add downsampling layers
            model += [nn.ELU()]
            model += [EncodecConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]
        model += [EncodecConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.CellList(model)

    def construct(self, hidden_states):
        """
        Constructs the encoded hidden states by applying each layer in the EncodecEncoder.

        Args:
            self (EncodecEncoder): An instance of the EncodecEncoder class.
            hidden_states (object):
                The input hidden states.

                - Type: Any valid Python object
                - Purpose: Represents the initial hidden states.
                - Restrictions: None

        Returns:
            None: This method does not return any value. It updates the hidden_states in place.

        Raises:
            None.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Cell):
    """SEANet decoder as used by EnCodec."""
    def __init__(self, config: EncodecConfig):
        """
        __init__

        Initializes an instance of the EncodecDecoder class.

        Args:
            self: The instance of the class.
            config (EncodecConfig):
                An instance of the EncodecConfig class containing configuration parameters for the decoder.

                - Type: EncodecConfig
                - Purpose: Specifies the configuration settings for the decoder.
                - Restrictions: Must be an instance of the EncodecConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [EncodecConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        model += [EncodecLSTM(config, scaling * config.num_filters)]

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add upsampling layers
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        model += [EncodecConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.CellList(model)

    def construct(self, hidden_states):
        """
        Construct method in the EncodecDecoder class.

        Args:
            self (object): Instance of the EncodecDecoder class.
            hidden_states (object): The hidden states to be processed by the method.
                This parameter is a list of hidden states that will be sequentially processed by each layer in the model.
                It is expected that each hidden state conforms to the input requirements of the layers.

        Returns:
            None: The method does not return any value directly but modifies the hidden_states in place.

        Raises:
            None.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Cell):
    """Codebook with Euclidean distance."""
    def __init__(self, config: EncodecConfig):
        """
        Initializes an instance of the EncodecEuclideanCodebook class.

        Args:
            self: The instance of the class.
            config (EncodecConfig): An object of the EncodecConfig class that contains the configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        embed = mindspore.Parameter(ops.zeros(config.codebook_size, config.codebook_dim), requires_grad=False)

        self.codebook_size = config.codebook_size

        self.inited = mindspore.Parameter([True], requires_grad=False)
        self.cluster_size = mindspore.Parameter(ops.zeros(config.codebook_size), requires_grad=False)
        self.embed = embed
        self.embed_avg = embed.clone()

    def quantize(self, hidden_states):
        """
        Quantizes the given hidden states using the Euclidean codebook encoding method.

        Args:
            self (EncodecEuclideanCodebook): An instance of the EncodecEuclideanCodebook class.
            hidden_states (Tensor): A tensor representing the hidden states to be quantized.

        Returns:
            None.

        Raises:
            None.
        """
        embed = self.embed.t()
        scaled_states = hidden_states.pow(2).sum(1, keepdims=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdims=True))
        embed_ind = dist.max(axis=-1, return_indices=True)[1]
        return embed_ind

    def encode(self, hidden_states):
        """
        Encodes the hidden states using the Euclidean Codebook method.

        Args:
            self: An instance of the EncodecEuclideanCodebook class.
            hidden_states (ndarray): A numpy array containing the hidden states to be encoded.
                The shape of the array is expected to be (batch_size, sequence_length, hidden_size).

        Returns:
            ndarray: A numpy array containing the encoded indices.
                The shape of the array is the same as the input hidden_states, except for the last dimension
                which is reduced to represent the indices of the codebook.

        Raises:
            None.
        """
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        """
        Decodes an embedding index using the Euclidean codebook method.

        Args:
            self (EncodecEuclideanCodebook): An instance of the EncodecEuclideanCodebook class.
            embed_ind (int): The index of the embedding to decode.

        Returns:
            None.

        Raises:
            None.
        """
        quantize = embedding(embed_ind, self.embed)
        return quantize


class EncodecVectorQuantization(nn.Cell):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """
    def __init__(self, config: EncodecConfig):
        """
        Initializes an instance of the EncodecVectorQuantization class.

        Args:
            self: The instance of the EncodecVectorQuantization class.
            config (EncodecConfig):
                An object of the EncodecConfig class that contains the configuration data for the vector quantization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        """
        Method to encode hidden states using vector quantization.

        Args:
            self (EncodeVectorQuantization): The instance of the EncodeVectorQuantization class.
            hidden_states (torch.Tensor):
                The hidden states to be encoded. Should be in the shape of (batch_size, hidden_dim, sequence_length).

        Returns:
            embed_in (torch.Tensor): The encoded representation of the hidden states.

        Raises:
            None.
        """
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        """
        Decode the embedded indices to obtain the quantized vectors.

        Args:
            self (EncodecVectorQuantization): The instance of the EncodecVectorQuantization class.
            embed_ind (Tensor): A 3D tensor containing the embedded indices.
                Its shape should be (batch_size, num_channels, num_embeddings).

        Returns:
            quantize (Tensor): A 3D tensor representing the quantized vectors after decoding.
                The shape of the tensor is (batch_size, num_embeddings, num_channels).

        Raises:
            ValueError: If the embed_ind tensor is not of the expected shape.
            RuntimeError: If there is an issue with decoding the embedded indices.
        """
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class EncodecResidualVectorQuantizer(nn.Cell):
    """Residual Vector Quantizer."""
    def __init__(self, config: EncodecConfig):
        """
        Initializes an instance of the EncodecResidualVectorQuantizer class.

        Args:
            self: The instance of the class.
            config (EncodecConfig):
                An object of the EncodecConfig class that holds configuration parameters.

                - codebook_size (int): The size of the codebook.
                - frame_rate (int): The frame rate.
                - num_quantizers (int): The number of quantizers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = config.num_quantizers
        self.layers = nn.CellList([EncodecVectorQuantization(config) for _ in range(config.num_quantizers)])

    def get_num_quantizers_for_bandwidth(self, bandwidth: Optional[float] = None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: mindspore.Tensor, bandwidth: Optional[float] = None) -> mindspore.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = ops.stack(all_indices)
        return out_indices

    def decode(self, codes: mindspore.Tensor) -> mindspore.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized_out = mindspore.tensor(0.0)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = EncodecConfig
    base_model_prefix = "encodec"
    main_input_name = "input_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(
                Normal(sigma=self.config.initializer_range, mean=0.0)))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros'))
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.beta.set_data(initializer('zeros', shape=cell.beta.data.shape))
            cell.gamma.set_data(initializer('ones', shape=cell.gamma.data.shape))
        elif isinstance(cell, nn.Conv1d):
            cell.weight.set_data(initializer('he_normal',shape=cell.weight.shape))
            if cell.bias is not None:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(Uniform(k), shape=cell.bias.shape))
                # nn.init.uniform_(cell.bias, a=-k, b=k)
        elif isinstance(cell, nn.Embedding):
            cell.embedding_table.set_data(initializer(
                Normal(sigma=self.config.initializer_range,mean=0.0)))
            #.normal_(mean=0.0, std=self.config.initializer_range)
            if cell.padding_idx is not None:
                cell.embedding_table.set_data(initializer('zeros',cell.padding_idx))
        elif isinstance(cell, nn.LSTM):
            for name, param in cell.parameters_and_names():
                if "weight" in name:
                    param.set_data(initializer('xavier_uniform',shape=param.shape))
                    # nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.set_data(initializer('zeros',shape=param.shape))
                    # nn.init.constant_(param, 0.0)


class EncodecModel(EncodecPreTrainedModel):

    """
    EncodecModel

    This class represents an Encodec model for audio encoding and decoding. It is a subclass of EncodecPreTrainedModel.

    Attributes:
        config (EncodecConfig): The configuration instance used to initialize the model.
        encoder (EncodecEncoder): The encoder module of the model.
        decoder (EncodecDecoder): The decoder module of the model.
        quantizer (EncodecResidualVectorQuantizer): The quantizer module of the model.
        bits_per_codebook (int): The number of bits per codebook.
        post_init (method): A method called after the initialization of the model.

    Methods:
        get_encoder(): Returns the encoder module of the model.
        get_decoder(): Returns the decoder module of the model.
        _encode_frame(input_values, bandwidth, padding_mask): Encodes the given input using the underlying VQVAE.
        encode(input_values, padding_mask, bandwidth, return_dict): Encodes the input audio waveform into discrete codes.
        _linear_overlap_add(frames, stride): Applies linear overlap-add to the given frames.
        _decode_frame(codes, scale): Decodes the given codes into an output audio waveform.
        decode(audio_codes, audio_scales, padding_mask, return_dict): Decodes the given frames into an output audio waveform.
        construct(input_values, padding_mask, bandwidth, audio_codes, audio_scales, return_dict): Constructs the model.

    Example:
        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel
        ...
        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]
        ...
        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)
        ...
        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```
    """
    def __init__(self, config: EncodecConfig):
        """
        Initializes an instance of the EncodecModel class.

        Args:
            self: The instance of the EncodecModel class.
            config (EncodecConfig): The configuration object containing settings for the EncodecModel.
                This parameter is required and must be of type EncodecConfig.
                It specifies the configuration settings for the EncodecModel.

        Returns:
            None.

        Raises:
            ValueError: If the codebook_size specified in the config is not a power of 2.
                This exception is raised when the codebook_size is invalid.
        """
        super().__init__(config)
        self.config = config

        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        self.quantizer = EncodecResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        This method returns the encoder associated with the EncodecModel instance.

        Args:
            self (EncodecModel): The instance of the EncodecModel class.
                It is used to access the attributes and methods of the class.

        Returns:
            encoder: This method returns the encoder associated with the EncodecModel instance.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        This method returns the decoder object associated with the EncodecModel instance.

        Args:
            self (object): The instance of the EncodecModel class.
                It is used to access the attributes and methods of the class.

        Returns:
            None: This method does not return any value explicitly, as it directly retrieves and returns the decoder
                object associated with the instance of the EncodecModel class.

        Raises:
            None.
        """
        return self.decoder

    def _encode_frame(
        self, input_values: mindspore.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the input is first
        normalized. The padding mask is required to compute the correct scale.
        """
        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate

        if self.config.chunk_length_s is not None and duration > 1e-5 + self.config.chunk_length_s:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}")

        scale = None
        if self.config.normalize:
            # if the padding is non zero
            input_values = input_values * padding_mask
            mono = ops.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            scale = mono.pow(2).mean(axis=-1, keep_dims=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        codes = codes.swapaxes(0, 1)
        return codes, scale

    def encode(
        self,
        input_values: mindspore.Tensor,
        padding_mask: mindspore.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, Optional[mindspore.Tensor]], EncodecEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`mindspore.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`mindspore.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = ops.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset : offset + chunk_length].bool()
            frame = input_values[:, :, offset : offset + chunk_length]
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = ops.stack(encoded_frames)

        if not return_dict:
            return (encoded_frames, scales)

        return EncodecEncoderOutput(encoded_frames, scales)

    @staticmethod
    def _linear_overlap_add(frames: List[mindspore.Tensor], stride: int):
        """
        Method _linear_overlap_add in the EncodecModel class.

        This method performs linear overlap-add method on a list of frames to reconstruct the original signal.

        Args:
            frames (List[mindspore.Tensor]): A list of mindspore tensors representing the input frames.
                Each frame should be a tensor of shape [batch_size, ... , frame_length].
            stride (int): An integer specifying the stride for overlapping frames.
                It determines the amount of overlap between consecutive frames.

        Returns:
            None: The method modifies the frames in-place to perform the linear overlap-add operation.

        Raises:
            ValueError:
                - If the input list of frames is empty.
                - If the minimum element of the sum of weights (sum_weight) is zero, indicating an invalid operation.
        """
        # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
        # e.g., more than 2 frames per position.
        # The core idea is to use a weight function that is a triangle,
        # with a maximum value at the middle of the chunk.
        # We use this weighting when summing the frames, and divide by the sum of weights
        # for each positions at the end. Thus:
        #   - if a frame is the only one to cover a position, the weighting is a no-op.
        #   - if 2 frames cover a position:
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       , i.e. S offset of second frame starts, T end of first frame.
        # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
        # After the final normalization, the weight of the second frame at position `t` is
        # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
        #
        #   - if more than 2 frames overlap at a given point, we hope that by induction
        #      something sensible happens.
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

        frame_length = frames[0].shape[-1]
        time_vec = ops.linspace(0, 1, frame_length + 2).to(dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        sum_weight = ops.zeros(total_size, dtype=dtype)
        out = ops.zeros(*shape, total_size, dtype=dtype)
        offset: int = 0

        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        if sum_weight.min() == 0:
            raise ValueError(f"`sum_weight` minimum element must be bigger than zero: {sum_weight}`")

        return out / sum_weight

    def _decode_frame(self, codes: mindspore.Tensor, scale: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        This method decodes the input codes and returns the corresponding output tensor.

        Args:
            self (EncodecModel): The instance of the EncodecModel class.
            codes (mindspore.Tensor): The input tensor containing the codes to be decoded.
                It is expected to have the shape (sequence_length, batch_size, code_size).
            scale (Optional[mindspore.Tensor]): An optional tensor representing the scale factor.
                If provided, it is expected to have the shape (batch_size, 1, 1). Defaults to None.

        Returns:
            mindspore.Tensor: The output tensor representing the decoded frames.
                It has the shape (sequence_length, batch_size, feature_size).

        Raises:
            ValueError: If the input codes or scale tensor have incompatible shapes for the decoding operation.
            TypeError: If the input codes or scale are not of type mindspore.Tensor.
        """
        codes = codes.swapaxes(0, 1)
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    def decode(
        self,
        audio_codes: mindspore.Tensor,
        audio_scales: mindspore.Tensor,
        padding_mask: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`mindspore.Tensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`mindspore.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`mindspore.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict or self.config.return_dict

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []

            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    def construct(
        self,
        input_values: mindspore.Tensor,
        padding_mask: Optional[mindspore.Tensor] = None,
        bandwidth: Optional[float] = None,
        audio_codes: Optional[mindspore.Tensor] = None,
        audio_scales: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], EncodecOutput]:
        r"""
        Returns:
            Union[Tuple[mindspore.Tensor, mindspore.Tensor], EncodecOutput]

        Example:
            ```python
            >>> from datasets import load_dataset
            >>> from transformers import AutoProcessor, EncodecModel
            ...
            >>> dataset = load_dataset("ashraq/esc50")
            >>> audio_sample = dataset["train"]["audio"][0]["array"]
            ...
            >>> model_id = "facebook/encodec_24khz"
            >>> model = EncodecModel.from_pretrained(model_id)
            >>> processor = AutoProcessor.from_pretrained(model_id)
            ...
            >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> audio_codes = outputs.audio_codes
            >>> audio_values = outputs.audio_values
            ```
        """
        return_dict = return_dict or self.config.return_dict

        if padding_mask is None:
            padding_mask = ops.ones_like(input_values).bool()

        if audio_codes is not None and audio_scales is None:
            raise ValueError("You specified `audio_codes` but did not specify the `audio_scales`")

        if audio_scales is not None and audio_codes is None:
            raise ValueError("You specified `audio_scales` but did not specify the `audio_codes`")

        if audio_scales is None and audio_codes is None:
            audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, False)

        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]
        if not return_dict:
            return (audio_codes, audio_values)

        return EncodecOutput(audio_codes=audio_codes, audio_values=audio_values)

__all__ =  [
    "EncodecModel",
    "EncodecPreTrainedModel",
]
