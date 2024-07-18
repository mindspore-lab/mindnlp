# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore ConvNext model."""

from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Normal

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from ...backbone_utils import BackboneMixin
from .configuration_convnext import ConvNextConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnext-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


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
    random_tensor = random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->ConvNext
class ConvNextDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initializes an instance of the ConvNextDropPath class.
        
        Args:
            self (object): The instance of the ConvNextDropPath class.
            drop_prob (Optional[float]): The probability of dropping a connection during training. 
                If not provided, defaults to None. Should be a float value between 0 and 1, inclusive.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct a drop path operation on the hidden states.
        
        Args:
            self (ConvNextDropPath): The instance of the ConvNextDropPath class.
            hidden_states (mindspore.Tensor):
                The input tensor of hidden states on which the drop path operation will be performed.
        
        Returns:
            mindspore.Tensor: The tensor resulting from applying the drop path operation on the input hidden states.
        
        Raises:
            ValueError: If the drop probability is not within the valid range.
            TypeError: If the input hidden_states is not a valid tensor type.
            RuntimeError: If the operation fails due to an internal error.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Method to generate a string representation of the drop probability in the ConvNextDropPath class.
        
        Args:
            self: ConvNextDropPath object. Represents the instance of the ConvNextDropPath class.
            
        Returns:
            str: A string representing the drop probability of the ConvNextDropPath object.
        
        Raises:
            None.
        """
        return "p={}".format(self.drop_prob)


class ConvNextLayerNorm(nn.Cell):
    r"""
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        Initializes an instance of the ConvNextLayerNorm class.

        Args:
            self: The object itself.
            normalized_shape (tuple): The shape of the input tensor normalized over the specified axes.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-06.
            data_format (str, optional): The format of the input data. Must be either 'channels_last' or 'channels_first'.
                Defaults to 'channels_last'.

        Returns:
            None

        Raises:
            NotImplementedError: If the data format is not supported.

        """
        super().__init__()
        self.weight = Parameter(ops.ones(normalized_shape))
        self.bias = Parameter(ops.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        epsilon=eps)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the ConvNextLayerNorm.

        Args:
            self (ConvNextLayerNorm): An instance of the ConvNextLayerNorm class.
            x (mindspore.Tensor): The input tensor to be normalized.

        Returns:
            mindspore.Tensor: The normalized tensor.

        Raises:
            TypeError: If the input tensor is not of type mindspore.Tensor.
            ValueError: If the data format is not 'channels_last' or 'channels_first'.
            ValueError: If the input tensor has an unsupported dtype.
        """
        if self.data_format == "channels_last":
            x, _, _ = self.layer_norm(x, self.weight, self.bias)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keep_dims=True)
            s = (x - u).pow(2).mean(1, keep_dims=True)
            x = (x - u) / ops.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNextEmbeddings(nn.Cell):
    """
    This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """
    def __init__(self, config):
        """
        Initializes the ConvNextEmbeddings class.

        Args:
            self: The instance of the ConvNextEmbeddings class.
            config: An object containing the configuration parameters for the ConvNextEmbeddings class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size,
            pad_mode='valid', has_bias=True
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs embeddings from the input pixel values using the ConvNextEmbeddings class.

        Args:
            self (ConvNextEmbeddings): An instance of the ConvNextEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing pixel values with shape (batch_size, num_channels, height, width).
                The pixel values should align with the channel dimension specified in the configuration.

        Returns:
            mindspore.Tensor: A tensor representing the embeddings generated from the input pixel values.
                The embeddings have the same shape as the input pixel values.

        Raises:
            ValueError: If the number of channels in the input pixel values does not match the configured number of channels.
        """
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextLayer(nn.Cell):
    """
    This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """
    def __init__(self, config, dim, drop_path=0):
        '''
        Initializes the ConvNextLayer.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings.
            dim: An integer representing the dimension for convolution operation.
            drop_path: A float representing the dropout probability for drop path regularization.

        Returns:
            None.

        Raises:
            ValueError: If config.hidden_act is not found in ACT2FN.
            TypeError: If config.layer_scale_init_value is not a positive number.
        '''
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, pad_mode='pad', group=dim, has_bias=True)  # depthwise conv
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = ACT2FN[config.hidden_act]
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.layer_scale_parameter = (
            Parameter(config.layer_scale_init_value * ops.ones((dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )
        self.drop_path = ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        '''
        Construct method in the ConvNextLayer class.

        Args:
            self: ConvNextLayer instance.
            hidden_states (mindspore.Tensor): The input hidden states tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying the convolutional layer operations.

        Raises:
            None.
        '''
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextStage(nn.Cell):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """
    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        """
        Initializes a ConvNextStage object with the provided configuration.

        Args:
            self (ConvNextStage): The ConvNextStage object itself.
            config (any): The configuration settings for the ConvNextStage.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 2.
            stride (int, optional): The stride of the convolution operation. Defaults to 2.
            depth (int): The depth of the ConvNextStage.
            drop_path_rates (list, optional): A list of dropout rates for each layer in the stage. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If in_channels is not equal to out_channels or stride is greater than 1.
            TypeError: If drop_path_rates is not a list.
        """
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.SequentialCell(
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='valid', has_bias=True),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.SequentialCell(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the next stage of a convolutional neural network.

        Args:
            self (ConvNextStage): An instance of the ConvNextStage class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.

        Returns:
            mindspore.Tensor: The tensor representing the output hidden states after the next stage.

        Raises:
            None.
        """
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states


class ConvNextEncoder(nn.Cell):

    """ConvNextEncoder is a Python class that represents an encoder for a Convolutional Neural Network (CNN) model.

    This class inherits from the nn.Cell class, which is a base class for all neural network layers in the MindSpore framework.

    The ConvNextEncoder class initializes a list of stages, where each stage consists of a ConvNextStage module.
    The number of stages is defined by the config.num_stages attribute. Each stage performs convolutional operations
    with different parameters, such as input and output channels, stride, and depth.
    The drop_path_rates parameter specifies the drop path rates for each stage.

    The construct method of the ConvNextEncoder class takes a tensor of hidden states as input and performs the forward
    pass through each stage. It optionally returns a tuple containing all hidden states at each stage, as specified by
    the output_hidden_states parameter.
    If return_dict is set to True, it returns an instance of the BaseModelOutputWithNoAttention class, which
    encapsulates the last hidden state and all hidden states.

    Note that this docstring is generated based on the provided code, and the actual implementation may contain
    additional methods or attributes.

    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvNextEncoder class.

        Args:
            self (ConvNextEncoder): The instance of the ConvNextEncoder class.
            config:
                A configuration object containing various settings for the ConvNextEncoder.

                - drop_path_rate (float): The rate at which to apply drop path regularization.
                - depths (list[int]): List of integers representing the depths of each stage.
                - hidden_sizes (list[int]): List of integers representing the number of hidden units in each stage.
                - num_stages (int): The total number of stages in the ConvNextEncoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.stages = nn.CellList()
        drop_path_rates = [
            x.tolist() for x in ops.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
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
        """
        Constructs the encoder for the ConvNext model.

        Args:
            self (ConvNextEncoder): The instance of the ConvNextEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed by the encoder.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states for each layer.
                Defaults to False.
            return_dict (Optional[bool], optional): Whether to return the output as a dictionary. Defaults to True.

        Returns:
            Union[Tuple, BaseModelOutputWithNoAttention]:
                The output value which can be a tuple of hidden states or BaseModelOutputWithNoAttention object.

        Raises:
            None
        """
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


class ConvNextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"
    _no_split_modules = ["ConvNextLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Dense, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.initialize(Normal(self.config.initializer_range))
            if module.bias is not None:
                module.bias.initialize('zeros')
        elif isinstance(module, nn.LayerNorm):
            module.bias.initialize('zeros')
            module.weight.initialize('ones')


class ConvNextModel(ConvNextPreTrainedModel):

    """
    The ConvNextModel class represents a ConvNext model for image processing tasks.
    It inherits from ConvNextPreTrainedModel and includes methods for model initialization and construction.

    The __init__ method initializes the ConvNextModel with the provided configuration.
    It sets up the embeddings, encoder, and layer normalization based on the configuration parameters.

    The construct method processes the input pixel values using the embeddings and encoder, and returns
    the last hidden state and pooled output. It allows for customization of returning hidden states and outputs
    as specified in the configuration parameters.

    Note:
        This docstring is based on the provided code snippet and does not include complete signatures or any other code.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ConvNextModel class.

        Args:
            self: The instance of the ConvNextModel class.
            config: A dictionary containing configuration parameters for the model.

        Returns:
            None

        Raises:
            TypeError: If the provided config parameter is not a dictionary.
            ValueError: If the config parameter does not contain the required keys for initializing the model.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], epsilon=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: mindspore.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        """
        Constructs a ConvNextModel by processing the given pixel values.

        Args:
            self (ConvNextModel): The instance of the ConvNextModel class.
            pixel_values (mindspore.Tensor): The input pixel values. It should be a tensor.
            output_hidden_states (Optional[bool]): Whether or not to output hidden states. Defaults to None.
            return_dict (Optional[bool]): Whether or not to use a return dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]: The constructed ConvNextModel output.
                It can be either a tuple or an instance of BaseModelOutputWithPoolingAndNoAttention.

        Raises:
            ValueError: If pixel_values is not specified.

        Note:
            - If output_hidden_states is not provided, it defaults to the value specified in the configuration.
            - If return_dict is not provided, it defaults to the value specified in the configuration.
            - The returned value may contain the last hidden state, pooled output, and additional encoder outputs.

        """
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


class ConvNextForImageClassification(ConvNextPreTrainedModel):

    """ConvNextForImageClassification

    This class represents a Convolutional Neural Network (CNN) model for image classification using the
    ConvNext architecture. The model is designed for tasks such as single-label or multi-label classification
    and regression.
    It inherits from the ConvNextPreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels in the classification task.
        convnext (ConvNextModel): The ConvNext model used for feature extraction.
        classifier (nn.Dense or nn.Identity): The classifier layer for predicting the final output.

    Methods:
        construct(pixel_values, labels, output_hidden_states, return_dict)
            Constructs the ConvNextForImageClassification model.

    """
    def __init__(self, config):
        """
        __init__

        Initializes an instance of the ConvNextForImageClassification class.

        Args:
            self: The instance of the class.
            config:
                An instance of the configuration class containing the necessary parameters for model initialization.

                - Type: config
                - Purpose: To configure the model with specific settings and hyperparameters.
                - Restrictions: Must be an instance of the appropriate configuration class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)

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
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int32, mindspore.int64):
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
                loss = ops.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class ConvNextBackbone(ConvNextPreTrainedModel, BackboneMixin):

    """
    This class represents the ConvNext backbone used in a ConvNext model for image processing tasks.
    It inherits functionality from ConvNextPreTrainedModel and BackboneMixin.

    The ConvNextBackbone class initializes the backbone architecture with ConvNextEmbeddings and ConvNextEncoder
    components. It also sets up layer normalization for hidden states based on the specified configuration.
    The construct method processes input pixel values through the embeddings and encoder, optionally returning
    hidden states and feature maps. It handles the logic for outputting the desired information based
    on the configuration settings.

    Returns:
        BackboneOutput: A named tuple containing the feature maps and hidden states of the backbone.

    Example:
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ...
        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnext-tiny-224")
        ...
        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the ConvNextBackbone class.

        Args:
            self: The instance of the class.
            config: A configuration object containing the necessary parameters for initializing the backbone.
                It should have the following attributes:

                - hidden_sizes (list): A list of integers representing the hidden layer sizes.
                - channels (list): A list of integers representing the number of channels for each stage.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        super()._init_backbone(config)

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextLayerNorm(num_channels, data_format="channels_first")
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
            BackboneOutput

        Example:
            ```python
            >>> from transformers import AutoImageProcessor, AutoBackbone
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
            >>> model = AutoBackbone.from_pretrained("facebook/convnext-tiny-224")
            ...
            >>> inputs = processor(image, return_tensors="pt")
            >>> outputs = model(**inputs)
            ```
        """
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
    "ConvNextForImageClassification",
    "ConvNextModel",
    "ConvNextPreTrainedModel",
    "ConvNextBackbone",
]
