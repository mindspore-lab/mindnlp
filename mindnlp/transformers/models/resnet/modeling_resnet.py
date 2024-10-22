# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore ResNet model."""

from typing import Optional

import mindspore

from mindspore.common.initializer import initializer, HeNormal

from mindnlp.core import nn
from mindnlp.core.nn import functional as F
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
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"


class ResNetConvLayer(nn.Module):

    """
    The ResNetConvLayer class represents a convolutional layer used in the ResNet neural network architecture. 
    
    This class inherits from the nn.Module class and is designed to process input data through a series of operations
    including convolution, normalization, and activation.
    
    Attributes:
        convolution (nn.Conv2d): The convolutional layer used for feature extraction.
        normalization (nn.BatchNorm2d): The batch normalization layer used for normalizing the outputs of
            the convolutional layer.
        activation (nn.Identity or callable): The activation function applied to the normalized outputs.
    
    Methods:
        __init__:
            Initializes the ResNetConvLayer with the specified parameters.

        forward:
            Applies the convolutional layer, normalization, and activation to the input tensor and returns the processed tensor.
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        """
        Initializes a ResNetConvLayer object.

        Args:
            self (ResNetConvLayer): The instance of the ResNetConvLayer class.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            stride (int, optional): The stride of the convolutional kernel. Defaults to 1.
            activation (str, optional): The activation function to be applied. Defaults to 'relu'.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method 'forward' in the class 'ResNetConvLayer'.

        Args:
            self: Instance of the ResNetConvLayer class.
                Type: ResNetConvLayer
                Purpose: Represents the current instance of the ResNetConvLayer class.
                Restrictions: None.

            input: Input tensor for the convolution layer.
                Type: mindspore.Tensor
                Purpose: Represents the input tensor to be processed by the convolution layer.
                Restrictions: Should be a valid mindspore.Tensor.

        Returns:
            hidden_state:
                A tensor representing the processed output after passing through the convolution layer:

                - Type: mindspore.Tensor
                - Purpose: Represents the transformed tensor after passing through the convolution layer.

        Raises:
            None.
        """
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """
    def __init__(self, config: ResNetConfig):
        """
        Initializes an instance of the ResNetEmbeddings class.

        Args:
            self: The instance of the ResNetEmbeddings class.
            config (ResNetConfig):
                The configuration object that contains parameters for the ResNet embeddings.

                - num_channels (int): The number of input channels.
                - embedding_size (int): The size of the output embeddings.
                - hidden_act (str): The activation function for the hidden layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.embedder = ResNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        forwards the embeddings for a given set of pixel values.

        Args:
            self (ResNetEmbeddings): An instance of the ResNetEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing the pixel values of an image.

        Returns:
            mindspore.Tensor: The embeddings generated from the pixel values.

        Raises:
            ValueError: If the number of channels in the pixel_values tensor does not match the number of channels
                set in the configuration.

        This method takes in the pixel values of an image and generates embeddings using the ResNet model.
        It first checks if the number of channels in the pixel_values tensor matches the number of channels
        set in the configuration. If they do not match, a ValueError is raised. Otherwise, the pixel_values tensor
        is passed through the embedder and then the pooler to generate the embeddings.
        The resulting embeddings are returned as a mindspore.Tensor object.
        """
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


class ResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        """
        Initializes a new instance of the ResNetShortCut class.

        Args:
            self: The object itself.
            in_channels (int): The number of input channels.
                This parameter specifies the number of channels in the input tensor.
                It must be a positive integer.
            out_channels (int): The number of output channels.
                This parameter specifies the number of channels produced by the convolution.
                It must be a positive integer.
            stride (int, optional): The stride of the convolution. Default is 2.
                This parameter determines the stride size of the convolution operation.
                It must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        forwards a hidden state tensor using convolution and normalization operations.

        Args:
            self (ResNetShortCut): The instance of the ResNetShortCut class.
            input (mindspore.Tensor): The input tensor for the forwardion process.

        Returns:
            mindspore.Tensor: A tensor representing the hidden state after applying convolution and normalization.

        Raises:
            None
        """
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu"):
        """
        Initializes a ResNetBasicLayer object with the specified parameters.

        Args:
            self: The object itself.
            in_channels (int): The number of input channels to the layer.
            out_channels (int): The number of output channels from the layer.
            stride (int, optional): The stride value for the layer. Defaults to 1.
            activation (str, optional): The type of activation function to apply. Defaults to 'relu'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvLayer(in_channels, out_channels, stride=stride),
            ResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        """
        forwards a ResNet basic layer by applying a series of operations to the input hidden state.

        Args:
            self (ResNetBasicLayer): An instance of the ResNetBasicLayer class.
            hidden_state: The input hidden state tensor. It should have the shape (batch_size, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        downsample_in_bottleneck: bool = False,
    ):
        """
        Initializes a ResNetBottleNeckLayer object.

        Args:
            self: The instance of the ResNetBottleNeckLayer class.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int, optional): The stride value for the convolutional layers. Defaults to 1.
            activation (str, optional): The activation function to be applied. Defaults to 'relu'.
            reduction (int, optional): The reduction factor for the number of output channels. Defaults to 4.
            downsample_in_bottleneck (bool, optional): Whether to downsample in the bottleneck layer. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvLayer(
                in_channels, reduces_channels, kernel_size=1, stride=stride if downsample_in_bottleneck else 1
            ),
            ResNetConvLayer(reduces_channels, reduces_channels, stride=stride if not downsample_in_bottleneck else 1),
            ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        """
        forwards a ResNet bottleneck layer.

        Args:
            self (ResNetBottleNeckLayer): An instance of the ResNetBottleNeckLayer class.
            hidden_state (Tensor): The input hidden state tensor.

        Returns:
            None.

        Raises:
            None.
        """
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """
    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        """
        Initializes a ResNetStage object.

        Args:
            self: The instance of the class.
            config (ResNetConfig): The configuration object for the ResNet model.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int, optional): The stride value for the convolutional layers. Defaults to 2.
            depth (int, optional): The depth of the ResNet stage. Defaults to 2.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not an instance of ResNetConfig.
            ValueError: If in_channels or out_channels are not integers, or if depth is not a positive integer.
        """
        super().__init__()

        layer = ResNetBottleNeckLayer if config.layer_type == "bottleneck" else ResNetBasicLayer

        if config.layer_type == "bottleneck":
            first_layer = layer(
                in_channels,
                out_channels,
                stride=stride,
                activation=config.hidden_act,
                downsample_in_bottleneck=config.downsample_in_bottleneck,
            )
        else:
            first_layer = layer(in_channels, out_channels, stride=stride, activation=config.hidden_act)
        self.layers = nn.Sequential(
            first_layer, *[layer(out_channels, out_channels, activation=config.hidden_act) for _ in range(depth - 1)]
        )

    def forward(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        forwards the hidden state of the ResNet stage based on the given input.

        Args:
            self (ResNetStage): An instance of the ResNetStage class.
            input (mindspore.Tensor): The input tensor for forwarding the hidden state.

        Returns:
            mindspore.Tensor: The forwarded hidden state tensor.

        Raises:
            None.
        """
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class ResNetEncoder(nn.Module):

    """
    ResNetEncoder is a class that represents a Residual Neural Network (ResNet) encoder.
    It is a subclass of nn.Module and is used for forwarding the encoder part of a ResNet model.

    Attributes:
        stages (nn.ModuleList): A list of ResNetStage instances representing the different stages of the ResNet encoder.

    Methods:
        __init__:
            Initializes a ResNetEncoder instance.

            Args:

            - config (ResNetConfig): An instance of ResNetConfig class containing the configuration parameters
            for the ResNet encoder.

        forward:
            forwards the ResNet encoder.

            Args:

            - hidden_state (mindspore.Tensor): The input hidden state tensor.
            - output_hidden_states (bool, optional): A flag indicating whether to output hidden states at each stage.
            Defaults to False.
            - return_dict (bool, optional): A flag indicating whether to return the output as a
            BaseModelOutputWithNoAttention instance. Defaults to True.

            Returns:

            - BaseModelOutputWithNoAttention: An instance of BaseModelOutputWithNoAttention containing the encoder output.

    """
    def __init__(self, config: ResNetConfig):
        """
        Initializes an instance of the ResNetEncoder class.

        Args:
            self: The current instance of the class.
            config (ResNetConfig): The configuration object specifying the parameters for the ResNetEncoder.
                It is expected to have the following attributes:

                - embedding_size (int): The size of the input embeddings.
                - hidden_sizes (List[int]): A list of integers specifying the number of output channels for each
                ResNet stage.
                - depths (List[int]): A list of integers specifying the number of residual blocks in each ResNet stage.
                - downsample_in_first_stage (bool): A boolean indicating whether to perform downsampling in the first
                 ResNet stage or not. If True, the stride is set to 2; otherwise, it is set to 1.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.stages = nn.ModuleList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: mindspore.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        """
        forwards the ResNetEncoder by processing the hidden state through the defined stages.

        Args:
            self (ResNetEncoder): The instance of the ResNetEncoder class.
            hidden_state (mindspore.Tensor): The input hidden state to be processed through the encoder.
            output_hidden_states (bool, optional): Whether to output hidden states at each stage. Defaults to False.
            return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to True.

        Returns:
            BaseModelOutputWithNoAttention: An instance of BaseModelOutputWithNoAttention containing the
                last hidden state and optionally all hidden states if output_hidden_states is set to True.

        Raises:
            ValueError: If hidden_state is not a valid mindspore.Tensor.
            TypeError: If hidden_state is not of type mindspore.Tensor.
            RuntimeError: If an error occurs during the processing of the hidden state.

        """
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    _no_split_modules = ["ResNetConvLayer", "ResNetShortCut"]
    _keys_to_ignore_on_load_unexpected = [r'num_batches_tracked']

    def _init_weights(self, module):
        """
        This method initializes the weights of the given module according to the specified initialization scheme.

        Args:
            self (ResNetPreTrainedModel): The instance of the ResNetPreTrainedModel class.
            module: The module for which the weights need to be initialized.

        Returns:
            None.

        Raises:
            TypeError: If the module is of an unsupported type.
            ValueError: If the module's weight and bias initialization fails.
        """
        if isinstance(module, nn.Conv2d):
            module.weight.assign_value(initializer(HeNormal(), module.weight.shape, module.weight.dtype))
            #module.weight.initialize(HeNormal(mode='fan_out', nonlinearity='relu'))
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.weight.assign_value(
                initializer(
                    "zeros",
                    module.bias.shape,
                    module.bias.dtype,
                )
            )
            module.weight.assign_value(
                initializer("ones", module.weight.shape, module.weight.dtype)
            )

class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: mindspore.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        """
        forwards a ResNet model.

        Args:
            self: The object instance.
            pixel_values (mindspore.Tensor): The input pixel values of the images.
                It should be a tensor of shape [batch_size, height, width, channels].
            output_hidden_states (Optional[bool]): Whether to return hidden states of the encoder. Defaults to None.
                If not provided, it uses the value from the configuration.
            return_dict (Optional[bool]): Whether to return the output as a dictionary. Defaults to None.
                If not provided, it uses the value from the configuration.

        Returns:
            BaseModelOutputWithPoolingAndNoAttention:
                An instance of the BaseModelOutputWithPoolingAndNoAttention class containing the following outputs:

                - last_hidden_state (mindspore.Tensor): The last hidden state of the encoder.
                It has a shape of [batch_size, sequence_length, hidden_size].
                - pooled_output (mindspore.Tensor): The pooled output of the encoder.
                It has a shape of [batch_size, hidden_size].
                - hidden_states (Tuple[mindspore.Tensor]): A tuple of hidden states of the encoder
                if `output_hidden_states` is set to True. Each hidden state is a tensor of shape
                [batch_size, sequence_length, hidden_size].

        Raises:
            None.

        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class ResNetForImageClassification(ResNetPreTrainedModel):

    """
    ResNetForImageClassification is a class that represents a ResNet model for image classification tasks.
    It inherits from the ResNetPreTrainedModel class and includes methods for initializing the model and
    performing image classification.

    Attributes:
        num_labels (int): The number of labels for the classification task.
        resnet (ResNetModel): The ResNet model used for feature extraction.
        classifier (nn.Sequential): The classifier module for final classification.
        config: Configuration settings for the model.

    Methods:
        __init__: Initializes the ResNetForImageClassification model with the given configuration.
        forward: forwards the model for image classification, taking pixel values, labels, and optional
            parameters as input and returning the classification output.

    Parameters:
        pixel_values (mindspore.Tensor, optional): Tensor containing the pixel values of the input images.
        labels (mindspore.Tensor, optional): Tensor containing the labels for computing classification/regression loss.
        output_hidden_states (bool, optional): Flag to indicate whether to return hidden states in the output.
        return_dict (bool, optional): Flag to indicate whether to return the output as a dictionary.

    Returns:
        ImageClassifierOutputWithNoAttention:
            An ImageClassifierOutputWithNoAttention object containing the classification output with optional loss value
            and hidden states.

    Notes:
        - Labels should be indices in the range [0, config.num_labels - 1].
        - Classification loss is computed using Cross-Entropy if config.num_labels > 1.
        - The problem type is automatically determined based on the number of labels and label data type.
    """
    def __init__(self, config):
        """
        Initializes the ResNetForImageClassification class.

        Args:
            self: The instance of the class.
            config (object): An object containing configuration parameters for the model.
                It should have the following attributes:

                - num_labels (int): The number of output labels.
                - hidden_sizes (list): A list of integers representing hidden layer sizes.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not an object.
            ValueError: If config.num_labels is not an integer or config.hidden_sizes is not a list.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.resnet = ResNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

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
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


class ResNetBackbone(ResNetPreTrainedModel, BackboneMixin):

    """
    ResNetBackbone

    This class represents a ResNet backbone for image processing tasks. It inherits from the ResNetPreTrainedModel
    and BackboneMixin classes.

    Attributes:
        num_features (List[int]): A list of integers representing the number of features in each
            hidden layer of the backbone.
        embedder (ResNetEmbeddings): An instance of the ResNetEmbeddings class used for embedding pixel values.
        encoder (ResNetEncoder): An instance of the ResNetEncoder class used for encoding the embedded features.
        stage_names (List[str]): A list of strings representing the names of the stages in the backbone.
        out_features (List[str]): A list of strings representing the names of the output features.
        config (object): An object containing the configuration parameters for the ResNetBackbone.

    Methods:
        __init__: Initializes the ResNetBackbone instance with the given configuration.
        forward: forwards the backbone and returns the output.
    """
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.embedding_size] + config.hidden_sizes
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)

        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: mindspore.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embedder(pixel_values)

        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )

__all__ = [
    "ResNetForImageClassification",
    "ResNetModel",
    "ResNetPreTrainedModel",
    "ResNetBackbone",
]
