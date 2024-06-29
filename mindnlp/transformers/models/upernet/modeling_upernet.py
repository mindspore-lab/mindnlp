# Copyright 2024 Huawei Technologies Co., Ltd
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
"""MindSpore UPerNet model."""

from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import CrossEntropyLoss

from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from .configuration_upernet import UPerNetConfig


def load_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then
    we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone
    model itself, then we load the pretrained backbone weights
    if specified.
    """
    from mindnlp.transformers import AutoBackbone, AutoConfig

    backbone_config = getattr(config, "backbone_config", None)
    use_timm_backbone = getattr(config, "use_timm_backbone", None)
    use_pretrained_backbone = getattr(config, "use_pretrained_backbone", None)
    backbone_checkpoint = getattr(config, "backbone", None)
    backbone_kwargs = getattr(config, "backbone_kwargs", None)
    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

    # If there is a backbone_config and a backbone checkpoint, and use_pretrained_backbone=False then the desired
    # behaviour is ill-defined: do you want to load from the checkpoint's config or the backbone_config?
    if backbone_config is not None and backbone_checkpoint is not None and use_pretrained_backbone is not None:
        raise ValueError("Cannot specify both config.backbone_config and config.backbone")

    # If any of the following are set, then the config passed in is from a model which contains a backbone.
    if (
            backbone_config is None
            and use_timm_backbone is None
            and backbone_checkpoint is None
            and backbone_checkpoint is None
    ):
        return AutoBackbone.from_config(config=config, **backbone_kwargs)

    # config from the parent model that has a backbone
    if use_timm_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_timm_backbone is True")
        # Because of how timm backbones were originally added to models, we need to pass in use_pretrained_backbone
        # to determine whether to load the pretrained weights.
        backbone = AutoBackbone.from_pretrained(
            backbone_checkpoint,
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            **backbone_kwargs,
        )
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_pretrained_backbone is True")
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            raise ValueError("Either config.backbone_config or config.backbone must be set")
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        backbone = AutoBackbone.from_config(config=backbone_config)
    return backbone


class UPerNetConvModule(nn.Cell):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int], str] = 0,
            bias: bool = False,
            dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            pad_mode='pad',
            has_bias=bias,
            dilation=dilation,
            weight_init='ones',
            bias_init='zeros'
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.activation(output)

        return output


class UPerNetPyramidPoolingBlock(nn.Cell):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UPerNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class UPerNetPyramidPoolingModule(nn.Cell):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = UPerNetPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.insert_child_to_cell(str(i), block)

    def construct(self, x: ms.Tensor) -> List[ms.Tensor]:
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = ops.interpolate(
                ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerNetHead(nn.Cell):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """

    def __init__(self, config, in_channels):
        super().__init__()

        self.config = config
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = in_channels
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1, has_bias=True, weight_init='ones',
                                    bias_init='zeros')

        # PSP Module
        self.psp_modules = UPerNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = UPerNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.CellList()
        self.fpn_convs = nn.CellList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = UPerNetConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = UPerNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UPerNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        pass
        # if isinstance(module, nn.Conv2d):
        #     module.weight.set_data(initializer(Normal(self.config.initializer_range, 0),
        #                                        shape=module.weight.shape, dtype=module.weight.dtype))
        #     if module.bias is not None:
        #         module.bias.set_data(initializer('zeros', shape=module.bias.shape, dtype=module.bias.dtype))

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = ops.cat(psp_outs, axis=1)
        output = self.bottleneck(psp_outs)

        return output

    def construct(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + ops.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = ops.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = ops.cat(fpn_outs, axis=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class UPerNetFCNHead(nn.Cell):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet]. It uses only the feature of a certain layer in encoder.

    Args:
        config:
            Configuration.
        in_index (int):
            Use the feature of which layer. Default: 2.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
            self, config, in_index: int = 2, kernel_size: int = 3,
            dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config
        self.in_channels = config.auxiliary_in_channels
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = [UPerNetConvModule(
            self.in_channels, self.channels, kernel_size=kernel_size,
            padding=conv_padding, dilation=dilation
        )]
        for i in range(self.num_convs - 1):
            convs.append(
                UPerNetConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.SequentialCell(*convs)
        if self.concat_input:
            self.conv_cat = UPerNetConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1, has_bias=True, weight_init='ones',
                                    bias_init='zeros')

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        pass
        # if isinstance(module, nn.Conv2d):
        #     module.weight.set_data(initializer(Normal(self.config.initializer_range, 0.0),
        #                                        shape=module.weight.shape, dtype=module.weight.dtype))
        #     if module.bias is not None:
        #         module.bias.set_data(initializer('zeros', shape=module.bias.shape, dtype=module.bias.dtype))

    def construct(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(ops.cat([hidden_states, output], axis=1))
        output = self.classifier(output)
        return output


class UPerNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UPerNetConfig
    main_input_name = "pixel_values"
    _no_split_modules = []

    def _init_weights(self, module):
        if isinstance(module, UPerNetPreTrainedModel):
            module.backbone.init_weights()
            module.decode_head.init_weights()
            if module.auxiliary_head is not None:
                module.auxiliary_head.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        self.backbone.init_weights()
        self.decode_head.init_weights()
        if self.auxiliary_head is not None:
            self.auxiliary_head.init_weights()


class UPerNetForSemanticSegmentation(UPerNetPreTrainedModel):
    """
        UPerNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.
    """

    def __init__(self, config):
        """
        Args:
            config ([`UPerNetConfig`]): Model configuration class with all the parameters of the model.
                Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        """
        super().__init__(config)

        self.backbone = load_backbone(config)

        # Semantic segmentation head(s)
        self.decode_head = UPerNetHead(config, in_channels=self.backbone.channels)
        self.auxiliary_head = UPerNetFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            pixel_values: Optional[ms.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[ms.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        Args:
            pixel_values (`mindspore.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
                returned tensors for more detail.
            labels (`mindspore.Tensor` of shape `(batch_size, height, width)`, *optional*):
                Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            mindnlp.transformers.modeling_outputs.SemanticSegmenterOutput or tuple(mindspore.Tensor)

        Examples:
        ```python

        >>> from mindnlp.transformers import AutoImageProcessor, UPerNetForSemanticSegmentation
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        >>> model = UPerNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
        >>> list(logits.shape)
        [1, 150, 512, 512]
        ```
        """
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states
        )
        features = outputs.feature_maps

        logits = self.decode_head(features)
        logits = ops.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
            auxiliary_logits = ops.interpolate(
                auxiliary_logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
            )

        loss = None
        if labels is not None:
            # compute weighted loss
            loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
            loss = loss_fct(logits, labels)
            if auxiliary_logits is not None:
                auxiliary_loss = loss_fct(auxiliary_logits, labels)
                loss += self.config.auxiliary_loss_weight * auxiliary_loss

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    'UPerNetForSemanticSegmentation',
]
