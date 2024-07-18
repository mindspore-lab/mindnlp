# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore ConvNextV2 model."""


from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...backbone_utils import BackboneMixin
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from .configuration_convnextv2 import ConvNextV2Config

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


#from ..deprecated._archive_maps import CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: mindspore.Tensor, drop_prob: float = 0.0, training: bool = False) -> mindspore.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->ConvNextV2
class ConvNextV2DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class ConvNextV2GRN(nn.Cell):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = Parameter(ops.zeros(1, 1, 1, dim), 'weight')
        self.bias = Parameter(ops.zeros(1, 1, 1, dim), 'bias')

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # Compute and normalize global spatial feature maps
        global_features = ops.norm(hidden_states, dim=(1, 2), keepdim =True)
        norm_features = global_features / (global_features.mean(axis =-1, keep_dims =True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states

        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->ConvNextV2
class ConvNextV2LayerNorm(nn.Cell):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = Parameter(ops.ones(normalized_shape), 'weight')
        self.bias = Parameter(ops.zeros(normalized_shape), 'bias')
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        epsilon=self.eps)
    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        if self.data_format == "channels_last":
            x, _, _ = self.layer_norm(x, self.weight, self.bias)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keep_dims =True)
            s = (x - u).pow(2).mean(1, keep_dims =True)
            x = (x - u) / ops.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextEmbeddings with ConvNext->ConvNextV2
class ConvNextV2Embeddings(nn.Cell):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size, pad_mode='valid', has_bias=True
        )
        self.layernorm = ConvNextV2LayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextV2Layer(nn.Cell):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, group=dim, pad_mode='pad', has_bias=True)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        x = x.permute(0, 2, 3, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextStage with ConvNeXT->ConvNeXTV2, ConvNext->ConvNextV2
class ConvNextV2Stage(nn.Cell):
    """ConvNeXTV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.SequentialCell(
                ConvNextV2LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='valid', has_bias=True),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.SequentialCell(
            *[ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextEncoder with ConvNext->ConvNextV2
class ConvNextV2Encoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.CellList()
        drop_path_rates = [
            x.tolist() for x in ops.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# Copied from transformers.models.convnext.modeling_convnext.ConvNextPreTrainedModel with ConvNext->ConvNextV2, convnext->convnextv2
class ConvNextV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextV2Config
    base_model_prefix = "convnextv2"
    main_input_name = "pixel_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


# Copied from transformers.models.convnext.modeling_convnext.ConvNextModel with CONVNEXT->CONVNEXTV2, ConvNext->ConvNextV2
class ConvNextV2Model(ConvNextV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm([config.hidden_sizes[-1]], epsilon=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()


    def construct(
        self,
        pixel_values: mindspore.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )



# Copied from transformers.models.convnext.modeling_convnext.ConvNextForImageClassification with CONVNEXT->CONVNEXTV2,ConvNext->ConvNextV2,convnext->convnextv2
class ConvNextV2ForImageClassification(ConvNextV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnextv2 = ConvNextV2Model(config)

        # Classifier head
        self.classifier = (
            nn.Dense(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()


    def construct(
        self,
        pixel_values: mindspore.Tensor = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convnextv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops. binary_cross_entropy_with_logitst(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )



# Copied from transformers.models.convnext.modeling_convnext.ConvNextBackbone with CONVNEXT->CONVNEXTV2,ConvNext->ConvNextV2,facebook/convnext-tiny-224->facebook/convnextv2-tiny-1k-224
class ConvNextV2Backbone(ConvNextV2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.CellDict(hidden_states_norms)

        # initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: mindspore.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnextv2-tiny-1k-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )

__all__ = [
        "ConvNextV2ForImageClassification",
        "ConvNextV2Model",
        "ConvNextV2PreTrainedModel",
        "ConvNextV2Backbone"
    ]
