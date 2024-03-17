# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Encodec Model
"""
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Tuple


import mindspore
from mindspore import nn, Tensor, Parameter, ops
from mindspore import log as logger
from mindspore import numpy as np
from mindspore.common.initializer import initializer, Normal

from ...modeling_utils import PreTrainedModel

from ...modeling_outputs import (
    ModelOutput
)

from .configuration_encodec import EncodecConfig

__all__ = [
    "EncodecModel",
    "EncodecConv1d",
    "EncodecConvTranspose1d",
    "EncodecLSTM",
    "EncodecResnetBlock",
]

ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "encodec_24khz",
    "encodec_48khz",
    # See all EnCodec models at https://huggingface.co/models?filter=encodec
]

def norm_except_dim(weight_v, pows, dim):
    r"""
    calculte g/||weight_v|| * weight_v method 
    """
    if dim == -1:
        return np.norm(weight_v, pows)
    if dim == 0:
        output_size = (weight_v.shape[0],) + (1,) * (weight_v.ndim - 1)
        return np.norm(weight_v.view((weight_v.shape[0], -1)), pows, 1).view(output_size)
    if dim == (weight_v.ndim - 1):
        output_size = (1,) * (weight_v.ndim - 1) + (weight_v.shape[weight_v.ndim - 1])
        return np.norm(weight_v.view((-1, weight_v.shape[weight_v.ndim - 1])), pows, 0).view(output_size)
    return norm_except_dim(weight_v.swapaxes(0, dim), pows, dim).swapaxes(0, dim)

def _weight_norm(weight_v, weight_g, dim):
    r"""
    calculte weight_g/||weight_v|| * weight_v method 
    """
    return weight_v * (weight_g / norm_except_dim(weight_v, 2, dim))

class WeightNorm:
    r"""
    Weight Normalization from https://arxiv.org/abs/1602.07868
    """
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, cell: nn.Cell) -> Any:
        r"""
        computer methods
        """
        weight_g = getattr(cell, self.name + '_g')
        weight_v = getattr(cell, self.name + '_v')
        return _weight_norm(weight_v=weight_v, weight_g=weight_g, dim=self.dim)

    def __call__(self, cell: nn.Cell, inputs: Any) -> None:
        setattr(cell, self.name, self.compute_weight(cell))

    def wrapper_func(self, cell, func):
        r"""
        wrapper_func where used to transpose cell_id to cell
        """
        def new_func(_, inputs):
            nonlocal cell
            return func(cell, inputs)
        return new_func

    @staticmethod
    def apply(cell: nn.Cell, name: str, dim: int) -> 'WeightNorm':
        r"""
        construct methods
        """
        # warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
        for hook in cell._forward_pre_hook.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f"Cannot register two weight_norm hooks on the same parameter {name}")

        if dim is None:
            dim = -1

        weight_fn = WeightNorm(name, dim)

        weight = getattr(cell, name)
        del cell._params[name]
        cell.insert_param_to_cell(name + '_g', param= Parameter(norm_except_dim(weight,2,dim)))
        cell.insert_param_to_cell(name + '_v', param= Parameter(weight.data))
        setattr(cell, name, Parameter(weight_fn.compute_weight(cell)))
        cell.register_forward_pre_hook(weight_fn.wrapper_func(cell, weight_fn.__call__))
        return weight_fn

    def remove(self, cell: nn.Cell) -> None:
        r"""
        remove weight bias
        """
        weight = self.compute_weight(cell)
        delattr(cell, self.name)
        del cell._parameters[self.name + '_g']
        del cell._parameters[self.name + '_v']
        setattr(cell, self.name, Parameter(weight.data))

def weight_norm(cell: nn.Cell, name: str = 'weight', dim: int = 0) -> nn.Cell:
    r"""Applies weight normalization to a parameter in the given cell.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~cell.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at cell forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the cell
          in question.

    Args:
        cell (cell): containing cell
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original cell with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(cell, name, dim)
    return cell



def remove_weight_norm(cell: nn.Cell, name: str = 'weight') -> nn.Cell:
    r"""Removes the weight normalization reparameterization from a cell.

    Args:
        cell (cell): containing cell
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in cell._forward_pre_hook.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(cell)
            del cell._forward_pre_hook[k]
            return cell

    raise ValueError(f"weight_norm of '{name}' not found in {cell}")

@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`mindspore.Tensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
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

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation,has_bias=True,pad_mode='valid')
        if self.norm_type == "weight_norm":
            # self.conv.weight.set_data(initializer(init='HeNormal',shape=self.conv.weight.shape))
            self.conv = weight_norm(self.conv)
            # self.conv.weight.set_data(initializer(init='HeNormal',shape=self.conv.weight.shape))

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
        """Tiny wrapper around mindspore.ops.pad, just to allow for reflect padding on small input.
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
        padded = ops.pad(hidden_states, paddings, mode)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def construct(self, hidden_states):
        r"""
        construct method
        """

        kernel_size = self.conv.kernel_size[-1]
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
            #Attention: beause of mindspore can't assistant to besides four diameters input ,so here deal by
            #sepecial method
            inputs = hidden_states.unsqueeze(-1)
            hidden_states = self.norm(inputs)
            hidden_states = hidden_states.squeeze(-1)

        return hidden_states

class EncodecConvTranspose1d(nn.Cell):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
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
        r"""
        construct method
        """
        kernel_size = self.conv.kernel_size[-1]
        stride = self.conv.stride[-1]
        padding_total = kernel_size - stride

        hidden_states = self.conv(hidden_states)
        if self.norm_type == "time_group_norm":
            inputs = hidden_states.unsqueeze(-1)
            hidden_states = self.norm(inputs)
            hidden_states = hidden_states.squeeze(-1)

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
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, config.num_lstm_layers)

    def construct(self, hidden_states):
        hidden_states = hidden_states.permute(2, 0, 1)
        hidden_states = self.lstm(hidden_states)[0] + hidden_states
        hidden_states = hidden_states.permute(1, 2, 0)
        return hidden_states

class EncodecResnetBlock(nn.Cell):
    """
    Residual block from SEANet model as used by EnCodec.
    """

    def __init__(self, config: EncodecConfig, dim: int, dilations: List[int]):
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
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Cell):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
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
        r"""
        construct method
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class EncodecDecoder(nn.Cell):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
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
        r"""
        construct method
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class EncodecEuclideanCodebook(nn.Cell):
    """Codebook with Euclidean distance."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        embed = ops.zeros((config.codebook_size, config.codebook_dim))

        self.codebook_size = config.codebook_size

        self.inited = Parameter(Tensor([True], dtype=mindspore.float32),requires_grad=False)
        self.cluster_size = Parameter(Tensor(ops.zeros(config.codebook_size)),requires_grad=False)
        self.embed = Parameter(embed,requires_grad=False)
        self.embed_avg = Parameter(ops.deepcopy(embed),requires_grad=False)
        # self.register_buffer("inited", mindspore.Tensor([True]))
        # self.register_buffer("cluster_size", ops.zeros(config.codebook_size))
        # self.register_buffer("embed", embed)
        # self.register_buffer("embed_avg", embed.clone())

    def quantize(self, hidden_states):
        r"""
        quantize method
        """
        embed = self.embed.t()
        scaled_states = ops.sum(hidden_states.pow(2), dim=1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + ops.sum(embed.pow(2), dim=0 , keepdim=True))
        _,embed_ind = dist.max(axis = -1,return_indices = True)
        return embed_ind

    def encode(self, hidden_states):
        r"""
        encode method
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
        r"""
        decode method
        """
        # quantize = mindspore.ms_function.embedding(self.embed, embed_ind)
        embedding = nn.Embedding(vocab_size = self.embed.shape[0], embedding_size = self.embed.shape[1])
        embedding.weight.set_data(self.embed)
        quantize = embedding(embed_ind)
        return quantize


class EncodecVectorQuantization(nn.Cell):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        r"""
        encode method
        """
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        r"""
        decode method
        """
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize

class EncodecResidualVectorQuantizer(nn.Cell):
    """Residual Vector Quantizer."""

    def __init__(self, config: EncodecConfig):
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
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        # print(type(cell.weight))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(
                Normal(sigma=self.config.initializer_range,mean=0.0)))
            #cell.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros'))
                # cell.bias.data.zero_()
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm)):
            cell.beta.set_data(initializer('zeros',shape=cell.beta.data.shape))
            cell.gamma.set_data(initializer('ones',shape=cell.gamma.data.shape))
            #cell.bias.data.zero_()
            #cell.weight.data.fill_(1.0)
        elif isinstance(cell, nn.Conv1d):
            # print(cell)
            # print(type(cell.weight))
            # print(cell.weight.data)
            cell.weight.set_data(initializer(init='HeNormal',shape=cell.weight.shape))
            # nn.init.kaiming_normal_(cell.weight)
            if cell.bias is not None:
                k = math.sqrt(cell.group / (cell.in_channels * cell.kernel_size[0]))
                cell.bias.set_data(initializer(ops.uniform(cell.bias.shape, Tensor(-k), Tensor(k)), shape=cell.bias.shape))
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

    def _set_gradient_checkpointing(self, cell, value=False):
        if isinstance(cell, (EncodecEncoder, EncodecDecoder)):
            cell.gradient_checkpointing = value



class EncodecModel(EncodecPreTrainedModel):
    r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float and padded to the approriate length in order to be encoded using chunks
            of length self.chunk_length and a stride of `config.chunk_stride`.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Mask to avoid computing scaling factors on padding token indices (can we avoid computing conv on these+).
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            <Tip warning={true}>

             `padding_mask` should always be passed, unless the input was truncated or not padded. This is because in
             order to process tensors effectively, the input audio should be padded so that `input_length % stride =
             step` with `step = chunk_length-stride`. This ensures that all chunks are of the same shape

            </Tip>

        bandwidth (`float`, *optional*):
            The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
            bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            `bandwidth == 6.0`
        audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

    def __init__(self, config: EncodecConfig):
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
        r"""
        get_encoder method
        """
        return self.encoder

    def get_decoder(self):
        r"""
        get_decoder method
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
            scale = ops.pow(mono, 2).mean(axis=-1, keep_dims=True).sqrt() + 1e-8
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
        time_vec = ops.linspace(Tensor(0, dtype=dtype), 1, frame_length + 2)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        sum_weight = ops.zeros(total_size, dtype=dtype)
        out = ops.zeros((*shape, total_size), dtype=dtype)
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
            audio_codes (`mindspore.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
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

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
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
