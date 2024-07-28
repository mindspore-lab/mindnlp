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
""" MindSpore TimeSformer model."""


import collections
from typing import Optional, Tuple, Union

import mindspore
from mindnlp.core import nn, ops
from mindspore.common.initializer import initializer, TruncatedNormal

from mindnlp.utils import logging

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_timesformer import TimesformerConfig


logger = logging.get_logger(__name__)


class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, config):
        """
        Initializes an instance of the TimesformerPatchEmbeddings class.
        
        Args:
            self: The object instance.
            config:
                An object containing configuration parameters for TimesformerPatchEmbeddings.

                - image_size (int or tuple): The size of the input image. If an int is provided,
                it is assumed to be a square image.
                - patch_size (int or tuple): The size of each patch. If an int is provided,
                it is assumed to be a square patch.
                - num_channels (int): The number of input channels in the image.
                - hidden_size (int): The size of the hidden projection space.

        Returns:
            None

        Raises:
            None

        """
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size,
                                    stride=patch_size, pad_mode='valid', bias=True)

    def forward(self, pixel_values):
        '''
        forward method in TimesformerPatchEmbeddings class.

        This method forwards patch embeddings from the input pixel_values.

        Args:
            self (TimesformerPatchEmbeddings): The instance of the TimesformerPatchEmbeddings class.
            pixel_values (torch.Tensor): A 5-dimensional tensor representing the pixel values of the input images,
                with dimensions (batch_size, num_frames, num_channels, height, width).

        Returns:
            tuple:
                A tuple containing the following values:

                - embeddings (torch.Tensor): A 3-dimensional tensor representing the forwarded embeddings,
                with dimensions (batch_size * num_frames, patch_width, num_channels).
                - num_frames (int): The number of frames in the input pixel_values.
                - patch_width (int): The width of the patches in the forwarded embeddings.

        Raises:
            None
        '''
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)

        embeddings = self.projection(pixel_values)
        patch_width = embeddings.shape[-1]
        embeddings = embeddings.flatten(start_dim=2).swapaxes(1, 2)
        return embeddings, num_frames, patch_width


class TimesformerEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """
    def __init__(self, config):
        """
        Initialize the TimesformerEmbeddings instance with the given configuration.

        Args:
            config (object):
                An object containing the configuration parameters for the TimesformerEmbeddings.

                - hidden_size (int): The embedding dimension.
                - num_frames (int): The number of frames in the input.
                - hidden_dropout_prob (float): The dropout rate for the embeddings.
                - attention_type (str): The type of attention mechanism to be used. Can be 'space_only'
                or any other value.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        self.attention_type = attention_type
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        # Positional Embeddings
        self.cls_token = mindspore.Parameter(ops.zeros(1, 1, embed_dim))
        self.position_embeddings = mindspore.Parameter(ops.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != "space_only":
            self.time_embeddings = mindspore.Parameter(ops.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def forward(self, pixel_values):
        """
        Constructs the embeddings for the Timesformer model.

        Args:
            self: An instance of the TimesformerEmbeddings class.
            pixel_values (torch.Tensor): A tensor of shape (batch_size, num_frames, patch_height, patch_width)
                representing the input pixel values.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, total_patches, embedding_dim) representing the
                forwarded embeddings.

        Raises:
            None.

        This method takes in a tensor of pixel values and forwards the embeddings for the Timesformer model.
        The input tensor is expected to have dimensions (batch_size, num_frames, patch_height, patch_width).
        The method first computes patch embeddings using the `patch_embeddings` function. It then adds a special token
        (`cls_token`) to the embeddings. If the shape of the embeddings does not match the shape of the position
        embeddings, the method adjusts the position embeddings to match the shape of the embeddings.
        The method then adds the adjusted position embeddings to the embeddings. The embeddings are
        then passed through a dropout layer (`pos_drop`).

        If the attention type is not 'space_only', the method separates the `cls_token` from the embeddings, converts
        the embeddings to the desired shape, and adds time embeddings to the embeddings. The time embeddings are
        adjusted to match the number of frames in the input tensor. The adjusted time embeddings are added to the
        embeddings. The embeddings are then passed through a dropout layer (`time_drop`).
        Finally, the embeddings are reshaped and the `cls_token` is concatenated back to the embeddings.

        The method returns the forwarded embeddings.
        """
        batch_size = pixel_values.shape[0]

        # create patch embeddings
        embeddings, num_frames, patch_width = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.broadcast_to((embeddings.shape[0], -1, -1))
        embeddings = ops.cat((cls_tokens, embeddings), axis=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if embeddings.shape[1] != self.position_embeddings.shape[1]:
            position_embeddings = self.position_embeddings
            cls_pos_embed = position_embeddings[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = position_embeddings[0, 1:, :].unsqueeze(0).swapaxes(1, 2)
            patch_num = int(other_pos_embed.shape[2] ** 0.5)
            patch_height = embeddings.shape[1] // patch_width
            other_pos_embed = other_pos_embed.reshape(1, embeddings.shape[2], patch_num, patch_num)
            new_pos_embed = ops.interpolate(
                other_pos_embed, size=(patch_height, patch_width), mode="nearest"
            )
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.swapaxes(1, 2)
            new_pos_embed = ops.cat((cls_pos_embed, new_pos_embed), 1)
            embeddings = embeddings + new_pos_embed
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.pos_drop(embeddings)

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = embeddings[:batch_size, 0, :].unsqueeze(1)
            embeddings = embeddings[:, 1:]
            _, patch_height, patch_width = embeddings.shape
            embeddings = (
                embeddings.reshape(batch_size, num_frames, patch_height, patch_width)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * patch_height, num_frames, patch_width)
            )
            # Resizing time embeddings in case they don't match
            if num_frames != self.time_embeddings.shape[1]:
                time_embeddings = self.time_embeddings.swapaxes(1, 2)
                new_time_embeddings = ops.interpolate(time_embeddings, size=(num_frames), mode="nearest")
                new_time_embeddings = new_time_embeddings.swapaxes(1, 2)
                embeddings = embeddings + new_time_embeddings
            else:
                embeddings = embeddings + self.time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.view(batch_size, patch_height, num_frames, patch_width).reshape(
                batch_size, patch_height * num_frames, patch_width
            )
            embeddings = ops.cat((cls_tokens, embeddings), axis=1)

        return embeddings


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
    random_tensor = random_tensor.floor(random_tensor)  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TimeSformer
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initializes an instance of the TimeSformerDropPath class.

        Args:
            self: The instance of the class.
            drop_prob (Optional[float]): The probability of dropping a path during training. Default is None.
                A floating-point number between 0 and 1 representing the probability of dropping a path.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to apply drop path regularization to the hidden states.

        Args:
            self (TimeSformerDropPath): An instance of the TimeSformerDropPath class.
            hidden_states (mindspore.Tensor):
                The hidden states tensor to apply drop path regularization to.

                - Type: mindspore.Tensor
                - Purpose: Represents the intermediate hidden states in the model.
                - Restrictions: Should be a tensor compatible with the drop_path function.

        Returns:
            mindspore.Tensor:
                The hidden states tensor after applying drop path regularization.

                - Type: mindspore.Tensor
                - Purpose: Represents the modified hidden states with drop path regularization applied.

        Raises:
            None.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Returns a string representation of the TimeSformerDropPath object.

        Args:
            self: The current instance of the TimeSformerDropPath class.

        Returns:
            string:
                A string representing the TimeSformerDropPath object. The string is formatted as 'p={}' where '{}'
                is replaced with the value of the 'drop_prob' attribute of the current instance.

        Raises:
            None.
        """
        return "p={}".format(self.drop_prob)


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L57
class TimesformerSelfAttention(nn.Module):

    """
    This class represents the self-attention mechanism used in the Timesformer model. It is a subclass of nn.Module.

    Attributes:
        num_heads (int): The number of attention heads.
        scale (float): The scaling factor applied to the attention scores.
        qkv (nn.Linear): The fully connected layer used to compute the query, key, and value representations.
        attn_drop (nn.Dropout): The dropout layer applied to the attention scores.

    Methods:
        __init__: Initializes the TimesformerSelfAttention instance.
        forward: Applies self-attention mechanism to the input hidden states.

    """
    def __init__(self, config: TimesformerConfig):
        """
        Initializes a new instance of the TimesformerSelfAttention class.

        Args:
            self: The current object instance.
            config (TimesformerConfig): The configuration object for Timesformer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attention_dropout_prob)

    def forward(self, hidden_states, output_attentions: bool = False):
        """
        Constructs the self-attention mechanism within a Timesformer model.

        Args:
            self (TimesformerSelfAttention): The instance of the TimesformerSelfAttention class.
            hidden_states (torch.Tensor): The input hidden states to be processed by the self-attention mechanism.
                Expected shape is (batch_size, hidden_size, num_channels).
            output_attentions (bool, optional): Flag indicating whether to output attention probabilities.
                Default is False.

        Returns:
            tuple:
                A tuple containing the context layer tensor and optionally the attention probabilities tensor.

                - context_layer (torch.Tensor): The output context layer after applying the self-attention mechanism.
                Shape is (batch_size, hidden_size, num_channels).
                - attention_probs (torch.Tensor), optional: The attention probabilities tensor if 'output_attentions'
                is True. Shape is (batch_size, self.num_heads, hidden_size, hidden_size).

        Raises:
            ValueError: If the input hidden_states tensor does not have the expected shape.
            RuntimeError: If an error occurs during the Softmax calculation or reshaping operations.
            AttributeError: If an attribute error is encountered while accessing class properties.
        """
        batch_size, hidden_size, num_channels = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, hidden_size, 3, self.num_heads, num_channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_probs = (query @ key.swapaxes(-2, -1)) * self.scale
        attention_probs = nn.Softmax(axis=-1)(attention_probs)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (attention_probs @ value).swapaxes(1, 2).reshape(batch_size, hidden_size, num_channels)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """
    def __init__(self, config: TimesformerConfig) -> None:
        """
        Initializes a new instance of the TimesformerSelfOutput class.

        Args:
            self: The instance of the TimesformerSelfOutput class.
            config (TimesformerConfig): A configuration object containing parameters for the self output layer.
                It specifies the hidden size and dropout probability for the layer.
                It must be an instance of the TimesformerConfig class.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type TimesformerConfig.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the self output of the Timesformer model.

        Args:
            self (TimesformerSelfOutput): An instance of the TimesformerSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor representing the input to the self output layer.

        Returns:
            hidden_states (mindspore.Tensor): The output tensor after applying the self output layer operations.

        Raises:
            None.

        This method takes the hidden states tensor and applies the self output layer operations to it. It first applies
        a dense layer to transform the hidden states tensor. Then, it applies dropout to the transformed tensor to
        prevent overfitting. Finally, it returns the output tensor after the self output layer operations have been
        applied.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimeSformerAttention(nn.Module):

    """
    The TimeSformerAttention class represents the self-attention mechanism for the TimeSformer model. It inherits from
    the nn.Module class and is responsible for performing self-attention and generating attention-based outputs.

    The class contains the following methods:

    - __init__: Initializes the TimeSformerAttention instance with the provided configuration.
    - forward: Constructs the self-attention mechanism using the provided hidden states and optionally returns
    attention outputs.

    This class is an essential component of the TimeSformer model, providing the functionality for self-attention
    computations and output generation.
    """
    def __init__(self, config: TimesformerConfig) -> None:
        """
        Initializes a new instance of the TimeSformerAttention class.

        Args:
            self: The current instance of the TimeSformerAttention class.
            config (TimesformerConfig): The configuration object specifying the settings for the TimeSformerAttention.

        Returns:
            None.

        Raises:
            None.

        This method initializes the TimeSformerAttention class by setting up the attention and output layers.
        The 'self' parameter refers to the current instance of the class, while the 'config' parameter is an instance
        of the TimesformerConfig class that specifies the configuration settings for the TimeSformerAttention.

        The 'attention' attribute is assigned an instance of the TimesformerSelfAttention class, which handles the
        attention mechanism.
        The 'output' attribute is assigned an instance of the TimesformerSelfOutput class, which processes the
        attention output.

        Example:
            ```python
            >>> config = TimesformerConfig()
            >>> time_sformer_attention = TimeSformerAttention(config)
            ```
        """
        super().__init__()
        self.attention = TimesformerSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
        '''
        This method forwards the attention mechanism for the TimeSformer model.

        Args:
            self: This parameter refers to the instance of the class itself.
            hidden_states (mindspore.Tensor): The input hidden states on which the attention mechanism is applied.
            output_attentions (bool, optional): A flag indicating whether to return the attention outputs.
                Defaults to False.

        Returns:
            Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:

                - If output_attentions is True, returns a tuple containing the attention output tensor and the attention
                weights tensor.
                - If output_attentions is False, returns a tuple containing only the attention output tensor.

        Raises:
            ValueError: If the hidden_states tensor is not of the expected format or shape.
            TypeError: If the input arguments are not of the expected types.
        '''
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L39
class TimesformerIntermediate(nn.Module):

    """
    The TimesformerIntermediate class represents a component of the Timesformer model that performs intermediate
    computations on the input hidden states. This class inherits from the nn.Module class.

    Attributes:
        dense (nn.Linear): A dense layer used for transforming the input hidden states.
        dropout (nn.Dropout): A dropout layer with a dropout probability specified in the configuration.
        intermediate_act_fn (function): The activation function applied to the intermediate hidden states.

    Methods:
        __init__: Initializes the TimesformerIntermediate instance with the provided configuration.
        forward: Performs intermediate computations on the input hidden states and returns the result.
    """
    def __init__(self, config: TimesformerConfig) -> None:
        """
        Initializes a new instance of the TimesformerIntermediate class.

        Args:
            self: The instance of the TimesformerIntermediate class.
            config (TimesformerConfig): An instance of the TimesformerConfig class containing configuration parameters
                for the TimesformerIntermediate class. It specifies the hidden size, intermediate size, and hidden
                dropout probability.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type TimesformerConfig.
            KeyError: If the config.hidden_act is not a valid string key in the ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the intermediate representation for the Timesformer model.

        Args:
            self: An instance of the TimesformerIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.
                It should be a Tensor object containing the hidden states data required for intermediate representation
                forwardion.

        Returns:
            mindspore.Tensor: Returns a Tensor object representing the intermediate representation forwarded using
                the input hidden states data.

        Raises:
            ValueError: If the input hidden_states is not a valid mindspore.Tensor object.
            RuntimeError: If any error occurs during the intermediate representation forwardion process.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimesformerOutput(nn.Module):

    """
    TimesformerOutput represents the output of the Timesformer model, containing methods for processing hidden states.

    This class inherits from nn.Module and provides functionality for processing hidden states through dense and dropout
    layers.

    Attributes:
        dense (nn.Linear): A dense layer used for transforming hidden states.
        dropout (nn.Dropout): A dropout layer used for regularization.

    Methods:
        __init__: Initializes the TimesformerOutput object with the provided configuration.
        forward: Applies dense and dropout layers to the input hidden states.

    Example:
        ```python
        >>> config = TimesformerConfig(intermediate_size=1024, hidden_size=512, hidden_dropout_prob=0.1)
        >>> output = TimesformerOutput(config)
        >>> processed_states = output.forward(hidden_states)
        ```

    Note:
        The TimesformerOutput class is designed to work in conjunction with the Timesformer model for processing hidden
        states efficiently.
    """
    def __init__(self, config: TimesformerConfig) -> None:
        """
        Initializes a new instance of TimesformerOutput.

        Args:
            self: The instance of the class.
            config (TimesformerConfig): The configuration object for the Timesformer model,
                specifying the model's parameters.

                - Type: TimesformerConfig
                - Purpose: Specifies the configuration settings for the Timesformer model.
                - Restrictions: Must be an instance of TimesformerConfig.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method 'forward' is defined within the class 'TimesformerOutput' and is used to process the input
        'hidden_states' through a series of operations and return the resulting tensor.

        Args:
            self (TimesformerOutput): The instance of the TimesformerOutput class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states.
                It is expected to be of type mindspore.Tensor.

        Returns:
            mindspore.Tensor: The processed tensor after applying the dense and dropout operations.

        Raises:
            None: This method does not explicitly raise any exceptions. However, it is important to note that the
                operations performed within this method may raise exceptions related to tensor manipulation in
                the mindspore library.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Adapted from https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L89
class TimesformerLayer(nn.Module):

    """
    This class represents a Timesformer layer, which is a component of the Timesformer model. The Timesformer layer
    includes various operations such as attention, intermediate, and output layers. It supports different attention types,
    including divided space-time, space only, and joint space-time.

    The TimesformerLayer class inherits from the nn.Module class.

    Attributes:
        config (TimesformerConfig): The configuration object for the Timesformer model.
        attention_type (str): The type of attention used in the layer. Valid values are 'divided_space_time',
            'space_only', and 'joint_space_time'.
        drop_path (nn.Layer or nn.Identity): The dropout layer used for drop path regularization.
        attention (TimeSformerAttention): The attention module used in the layer.
        intermediate (TimesformerIntermediate): The intermediate module used in the layer.
        output (TimesformerOutput): The output module used in the layer.
        layernorm_before (nn.LayerNorm): The layer normalization module applied before the attention operation.
        layernorm_after (nn.LayerNorm): The layer normalization module applied after the attention operation.
        temporal_layernorm (nn.LayerNorm): The layer normalization module applied to temporal embeddings in case of
            'divided_space_time' attention type.
        temporal_attention (TimeSformerAttention): The attention module applied to temporal embeddings in case of
            'divided_space_time' attention type.
        temporal_dense (nn.Linear): The dense layer applied to temporal embeddings in case of 'divided_space_time'
            attention type.

    Methods:
        __init__:
            Initializes the TimesformerLayer object with the given configuration and layer index. Sets up the various
            modules and attributes of the layer.

        forward:
            Constructs the TimesformerLayer by applying the attention and intermediate operations to the given hidden
            states. Returns the output hidden states and optionally the attention outputs.

    Note:
        - The TimesformerLayer class assumes that the following modules are defined: TimeSformerDropPath,
        TimeSformerAttention, TimesformerIntermediate, and TimesformerOutput.
        - The forward method assumes that the following operations are defined: mindspore.Tensor.shape,
        ops.linspace, ops.cat, ops.mean, ops.reshape, ops.tile, and ops.permute.

    Raises:
        ValueError: If the provided attention type is not one of the valid options.
    """
    def __init__(self, config: TimesformerConfig, layer_index: int) -> None:
        """
        Initializes a TimesformerLayer instance.

        Args:
            self: The TimesformerLayer instance.
            config (TimesformerConfig): The configuration object for the Timesformer model.
            layer_index (int): The index of the layer.

        Returns:
            None.

        Raises:
            ValueError: If the attention_type in the config is not one of
                ['divided_space_time', 'space_only', 'joint_space_time'].
        """
        super().__init__()

        attention_type = config.attention_type

        drop_path_rates = [
            x.item() for x in ops.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        drop_path_rate = drop_path_rates[layer_index]

        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.config = config
        self.attention_type = attention_type
        if attention_type not in ["divided_space_time", "space_only", "joint_space_time"]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: mindspore.Tensor, output_attentions: bool = False):
        """
        Construct a Timesformer layer.

        Args:
            self (TimesformerLayer): The TimesformerLayer instance.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            None.

        Raises:
            ValueError: If the attention_type is not one of ['space_only', 'joint_space_time', 'divided_space_time'].
            ValueError: If the shape of the hidden_states tensor is not as expected.
            ValueError: If the shape of the temporal_embedding tensor is not as expected.
            ValueError: If the shape of the spatial_embedding tensor is not as expected.
            ValueError: If the shape of the cls_token tensor is not as expected.
            ValueError: If the shape of the residual_spatial tensor is not as expected.
            ValueError: If the shape of the hidden_states tensor after operations is not as expected.
            ValueError: If the shape of the layer_output tensor is not as expected.
            ValueError: If the shape of the outputs tuple is not as expected.
        """
        num_frames = self.config.num_frames
        num_patch_width = self.config.image_size // self.config.patch_size
        batch_size = hidden_states.shape[0]
        num_spatial_tokens = (hidden_states.shape[1] - 1) // num_frames
        num_patch_height = num_spatial_tokens // num_patch_width

        if self.attention_type in ["space_only", "joint_space_time"]:
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states), output_attentions=output_attentions
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            hidden_states = hidden_states + self.drop_path(attention_output)

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs

        elif self.attention_type == "divided_space_time":
            # Temporal
            temporal_embedding = hidden_states[:, 1:, :]
            temporal_embedding = temporal_embedding.reshape(
                batch_size, num_patch_height, num_patch_width, num_frames, temporal_embedding.shape[2]
            ).reshape(batch_size * num_patch_height * num_patch_width, num_frames, temporal_embedding.shape[2])

            temporal_attention_outputs = self.temporal_attention(
                self.temporal_layernorm(temporal_embedding),
            )
            attention_output = temporal_attention_outputs[0]

            residual_temporal = self.drop_path(attention_output)

            residual_temporal = residual_temporal.reshape(
                batch_size, num_patch_height, num_patch_width, num_frames, residual_temporal.shape[2]
            ).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_temporal.shape[2])
            residual_temporal = self.temporal_dense(residual_temporal)
            temporal_embedding = hidden_states[:, 1:, :] + residual_temporal

            # Spatial
            init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.tile((1, num_frames, 1))
            cls_token = cls_token.reshape(batch_size * num_frames, 1, cls_token.shape[2])
            spatial_embedding = temporal_embedding
            spatial_embedding = (
                spatial_embedding.reshape(
                    batch_size, num_patch_height, num_patch_width, num_frames, spatial_embedding.shape[2]
                )
                .permute(0, 3, 1, 2, 4)
                .reshape(batch_size * num_frames, num_patch_height * num_patch_width, spatial_embedding.shape[2])
            )
            spatial_embedding = ops.cat((cls_token, spatial_embedding), 1)

            spatial_attention_outputs = self.attention(
                self.layernorm_before(spatial_embedding), output_attentions=output_attentions
            )
            attention_output = spatial_attention_outputs[0]
            outputs = spatial_attention_outputs[1:]  # add self attentions if we output attention weights

            residual_spatial = self.drop_path(attention_output)

            # Taking care of CLS token
            cls_token = residual_spatial[:, 0, :]
            cls_token = cls_token.reshape(batch_size, num_frames, cls_token.shape[1])
            cls_token = ops.mean(cls_token, 1, True)  # averaging for every frame
            residual_spatial = residual_spatial[:, 1:, :]
            residual_spatial = (
                residual_spatial.reshape(
                    batch_size, num_frames, num_patch_height, num_patch_width, residual_spatial.shape[2]
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_spatial.shape[2])
            )
            residual = residual_spatial
            hidden_states = temporal_embedding

            # Mlp
            hidden_states = ops.cat((init_cls_token, hidden_states), 1) + ops.cat((cls_token, residual), 1)
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs


class TimesformerEncoder(nn.Module):

    """
    The TimesformerEncoder class represents a Timesformer encoder module that is used for encoding input sequences.
    It inherits from the nn.Module class.

    Attributes:
        config (TimesformerConfig): The configuration object that specifies the hyperparameters of the
            Timesformer encoder.
        layer (nn.ModuleList): A list of TimesformerLayer instances that make up the encoder's layers.

    Note:
        - The TimesformerEncoder is composed of multiple TimesformerLayer instances.
        - The hidden states and attention weights can be optionally returned for each layer.
        - The output can be returned either as a tuple or as a BaseModelOutput dictionary.
    """
    def __init__(self, config: TimesformerConfig) -> None:
        """
        Initializes the TimesformerEncoder object with the given configuration.

        Args:
            self (TimesformerEncoder): The instance of the TimesformerEncoder class.
            config (TimesformerConfig):
                The configuration object containing the settings for the TimesformerEncoder.

                - The 'config' parameter must be an instance of the TimesformerConfig class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TimesformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        # dady self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Constructs the TimesformerEncoder.

        Args:
            self (TimesformerEncoder): An instance of the TimesformerEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states. Expected shape is
                (batch_size, sequence_length, hidden_size).
            output_attentions (bool, optional): Whether to output the attention weights of each layer. Defaults to False.
            output_hidden_states (bool, optional): Whether to output the hidden states of each layer. Defaults to False.
            return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to True.

        Returns:
            Union[tuple, BaseModelOutput]: The output of the TimesformerEncoder.
                If return_dict is True, returns a dictionary with the following keys:

                - last_hidden_state (mindspore.Tensor): The last layer's hidden states. Shape is
                (batch_size, sequence_length, hidden_size).
                - hidden_states (tuple): A tuple containing the hidden states of each layer. Each hidden state has a
                shape of (batch_size, sequence_length, hidden_size).
                - attentions (tuple): A tuple containing the attention weights of each layer. Each attention weight
                matrix has a shape of (batch_size, num_heads, sequence_length, sequence_length).
            If return_dict is False, returns a tuple containing only the non-None values from the dictionary.

        Raises:
            None.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

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


class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TimesformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """
        Initializes the weights and biases of the given cell.

        Args:
            self: An instance of the TimesformerPreTrainedModel class.
            cell: A cell object to initialize weights and biases for.

        Returns:
            None: This method modifies the weights and biases of the given cell in-place.

        Raises:
            None.

        This method initializes the weights and biases of the given cell based on its type.
        It supports three types of cells: nn.Linear, nn.Conv2d, and nn.LayerNorm.

        For nn.Linear and nn.Conv2d cells:

        - The weights are initialized using the initializer function with a TruncatedNormal distribution with a
        standard deviation of self.config.initializer_range.
        - The biases are initialized with zeros using the initializer function.
        - If the cell does not have biases (cell.bias is None), no bias initialization is performed.

        For nn.LayerNorm cells:

        - The biases are initialized with zeros using the initializer function.
        - The weights are initialized with ones using the initializer function.

        For TimesformerEmbeddings cells:

        - The cls_token is initialized using the initializer function with a TruncatedNormal distribution with a standard
        deviation of self.config.initializer_range.
        - The position_embeddings are initialized using the initializer function with a TruncatedNormal distribution
        with a standard deviation of self.config.initializer_range.
        - The weights of the patch_embeddings are initialized by applying the _init_weights method recursively.

        Note:
            This method modifies the weights and biases of the given cell in-place.
        """
        if isinstance(cell, (nn.Linear, nn.Conv2d)):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=self.config.initializer_range), cell.weight.shape,
                                             cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, TimesformerEmbeddings):
            cell.cls_token.set_data(initializer(TruncatedNormal(sigma=self.config.initializer_range), cell.cls_token.shape,
                                                cell.cls_token.dtype))
            cell.position_embeddings.set_data(initializer(TruncatedNormal(sigma=self.config.initializer_range),
                                                          cell.position_embeddings.shape, cell.position_embeddings.dtype))
            cell.patch_embeddings.apply(self._init_weights)


class TimesformerModel(TimesformerPreTrainedModel):

    """
    Represents a Timesformer model.

    This class inherits from TimesformerPreTrainedModel and implements the Timesformer model architecture for
    processing video data. It includes methods for initializing the model, getting input embeddings, pruning heads,
    and forwarding the model output.

    The __init__ method initializes the TimesformerModel with the provided configuration. The get_input_embeddings
    method returns the patch embeddings from the model's embeddings. The _prune_heads method prunes heads of the model
    based on the provided heads_to_prune dictionary. The forward method processes the input pixel values and returns
    the model output, with options to include attentions and hidden states in the returned dictionary.

    The class also contains additional methods and attributes inherited from the base class TimesformerPreTrainedModel.

    Example usage and explanations are provided within the docstring for reference and clarity.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the TimesformerModel class.

        Args:
            self: An instance of the TimesformerModel class.
            config (dict): A dictionary containing configuration parameters for the model.
                This dictionary should include the necessary settings for configuring the model.
                Example fields include 'hidden_size', 'layer_norm_eps', etc.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of type dict.
            AttributeError: If any required field is missing in the config dictionary.
            ValueError: If the provided configuration settings are invalid or inconsistent.
            RuntimeError: If there are issues during the initialization process of the model components.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddings(config)
        self.encoder = TimesformerEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings for the TimesformerModel.

        Args:
            self: TimesformerModel
                The instance of the TimesformerModel class.

        Returns:
            None.

        Raises:
            None
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: mindspore.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutput]:
        r"""

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutput]:

        Example:
            ```python
            >>> import av
            >>> import numpy as np
            ...
            >>> from transformers import AutoImageProcessor, TimesformerModel
            >>> from huggingface_hub import hf_hub_download
            ...
            >>> np.random.seed(0)
            ...
            >>> def read_video_pyav(container, indices):
            ...     '''
            ...     Decode the video with PyAV decoder.
            ...     Args:
            ...         container (`av.container.input.InputContainer`): PyAV container.
            ...         indices (`List[int]`): List of frame indices to decode.
            ...     Returns:
            ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
            ...     '''
            ...     frames = []
            ...     container.seek(0)
            ...     start_index = indices[0]
            ...     end_index = indices[-1]
            ...     for i, frame in enumerate(container.decode(video=0)):
            ...         if i > end_index:
            ...             break
            ...         if i >= start_index and i in indices:
            ...             frames.append(frame)
            ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            ...
            ...
            >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
            ...     '''
            ...     Sample a given number of frame indices from the video.
            ...     Args:
            ...         clip_len (`int`): Total number of frames to sample.
            ...         frame_sample_rate (`int`): Sample every n-th frame.
            ...         seg_len (`int`): Maximum allowed index of sample's last frame.
            ...     Returns:
            ...         indices (`List[int]`): List of sampled frame indices
            ...     '''
            ...     converted_len = int(clip_len * frame_sample_rate)
            ...     end_idx = np.random.randint(converted_len, seg_len)
            ...     start_idx = end_idx - converted_len
            ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
            ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            ...     return indices
            ...
            ...
            >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
            >>> file_path = hf_hub_download(
            ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            ... )
            >>> container = av.open(file_path)
            ...
            >>> # sample 8 frames
            >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
            >>> video = read_video_pyav(container, indices)
            ...
            >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            >>> model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
            ...
            >>> # prepare video for the model
            >>> inputs = image_processor(list(video), return_tensors="pt")
            ...
            >>> # forward pass
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
            >>> list(last_hidden_states.shape)
            [1, 1569, 768]
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TimesformerForVideoClassification(TimesformerPreTrainedModel):

    """TimesformerForVideoClassification

    This class is a video classification model based on the Timesformer architecture.
    It inherits from the TimesformerPreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels for classification.
        timesformer (TimesformerModel): The Timesformer model for video classification.
        classifier (nn.Linear or nn.Identity): The classifier layer for the model.
        config (Config): The configuration object for the model.

    Methods:
        __init__: Initializes the TimesformerForVideoClassification instance.
        forward: Constructs the model and computes the loss and output.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the TimesformerForVideoClassification class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration parameters for the model.

                - num_labels (int): The number of output labels for classification.
                Must be a positive integer.
                - hidden_size (int): The size of the hidden layers in the model.
                Must be a positive integer.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' object does not contain the required attributes.
            TypeError: If the 'num_labels' or 'hidden_size' attributes in 'config' are not integers.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.timesformer = TimesformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        r"""

        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, ImageClassifierOutput]

        Example:
            ```python
            >>> import av
            >>> import mindspore
            >>> import mindnlp
            >>> import numpy as np
            ...
            >>> from mindnlp.transformers import AutoImageProcessor, TimesformerForVideoClassification
            >>> from huggingface_hub import hf_hub_download
            ...
            >>> np.random.seed(0)
            ...
            ...
            >>> def read_video_pyav(container, indices):
            ...     '''
            ...     Decode the video with PyAV decoder.
            ...     Args:
            ...         container (`av.container.input.InputContainer`): PyAV container.
            ...         indices (`List[int]`): List of frame indices to decode.
            ...     Returns:
            ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
            ...     '''
            ...     frames = []
            ...     container.seek(0)
            ...     start_index = indices[0]
            ...     end_index = indices[-1]
            ...     for i, frame in enumerate(container.decode(video=0)):
            ...         if i > end_index:
            ...             break
            ...         if i >= start_index and i in indices:
            ...             frames.append(frame)
            ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            ...
            ...
            >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
            ...     '''
            ...     Sample a given number of frame indices from the video.
            ...     Args:
            ...         clip_len (`int`): Total number of frames to sample.
            ...         frame_sample_rate (`int`): Sample every n-th frame.
            ...         seg_len (`int`): Maximum allowed index of sample's last frame.
            ...     Returns:
            ...         indices (`List[int]`): List of sampled frame indices
            ...     '''
            ...     converted_len = int(clip_len * frame_sample_rate)
            ...     end_idx = np.random.randint(converted_len, seg_len)
            ...     start_idx = end_idx - converted_len
            ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
            ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            ...     return indices
            ...
            ...
            >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
            >>> file_path = hf_hub_download(
            ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            ... )
            >>> container = av.open(file_path)
            ...
            >>> # sample 8 frames
            >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            >>> video = read_video_pyav(container, indices)
            ...
            >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            >>> model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
            ...
            >>> inputs = image_processor(list(video), return_tensors="ms")
            ...
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            ...
            >>> # model predicts one of the 400 Kinetics-400 classes
            >>> predicted_label = logits.argmax(-1).item()
            >>> print(model.config.id2label[predicted_label])
            eating spaghetti
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.timesformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in [mindspore.int64, mindspore.int32]):
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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "TimesformerForVideoClassification",
    "TimesformerPreTrainedModel",
    "TimesformerModel"
]
