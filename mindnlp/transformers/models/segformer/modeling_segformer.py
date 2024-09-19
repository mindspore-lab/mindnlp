# coding=utf-8
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore SegFormer model."""
import math
from typing import Optional, Tuple, Union

import mindspore

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import logging
from .configuration_segformer import SegformerConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "SegformerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


class SegFormerImageClassifierOutput(ImageClassifierOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Segformer
class SegformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initialize an instance of SegformerDropPath.
        
        Args:
            self: The instance of the SegformerDropPath class.
            drop_prob (Optional[float]): The probability of dropping a connection during training. 
                If None, no connections are dropped. Default is None.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a drop path operation on the hidden states.
        
        Args:
            self (SegformerDropPath): An instance of the SegformerDropPath class.
            hidden_states (mindspore.Tensor): The input hidden states to apply drop path on.
            
        Returns:
            mindspore.Tensor: The modified hidden states after applying drop path operation.
            
        Raises:
            TypeError: If the input hidden_states is not a mindspore.Tensor object.
            ValueError: If the drop_prob is not a valid probability value.
            
        Note:
            Drop path is a regularization technique used in training deep neural networks.
            It randomly sets a fraction of the hidden states to zero during training,
            which helps in reducing overfitting.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        This method returns a string representation of the drop probability for a SegformerDropPath instance.
        
        Args:
            self (SegformerDropPath):
                The instance of SegformerDropPath for which the drop probability is being represented.
        
        Returns:
            str: A string representing the drop probability of the SegformerDropPath instance.
        
        Raises:
            None.
        """
        return "p={}".format(self.drop_prob)


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""
    def __init__(self, patch_size, stride, num_channels, hidden_size):
        """Initialize the SegformerOverlapPatchEmbeddings class.
        
            Args:
                self: The object instance.
                patch_size (int): The size of the patches used for the convolutional layer.
                stride (int): The stride value for the convolutional layer.
                num_channels (int): The number of input channels.
                hidden_size (int): The number of output channels.
        
            Returns:
                None
        
            Raises:
                None
            """
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        """
        Constructs the overlap patch embeddings for the input pixel values.
        
        Args:
            self (SegformerOverlapPatchEmbeddings): An instance of the SegformerOverlapPatchEmbeddings class.
            pixel_values (torch.Tensor): A tensor representing the input pixel values.
                The shape of the tensor should be (batch_size, channels, height, width).
        
        Returns:
            tuple:
                A tuple containing the following elements:

                - embeddings (torch.Tensor): A tensor representing the forwarded embeddings.
                The shape of the tensor will be (batch_size, num_patches, embedding_dim).
                - height (int): The height of the embeddings tensor.
                - width (int): The width of the embeddings tensor.

        Raises:
            None.

        Note:
            - The 'proj' method referred to in the code should be a method defined in the
            SegformerOverlapPatchEmbeddings class.
            - The 'layer_norm' method referred to in the code should be a method defined in the
            SegformerOverlapPatchEmbeddings class.
        """
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = embeddings.flatten(start_dim=2).swapaxes(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        """Initializes an instance of the SegformerEfficientSelfAttention class.

        Args:
            self: The instance of the class.
            config: Configuration object containing various settings.
            hidden_size (int): The size of the hidden states.
            num_attention_heads (int): The number of attention heads.
            sequence_reduction_ratio (int): The ratio by which the sequence length is reduced.

        Returns:
            None

        Raises:
            ValueError: If the hidden_size is not a multiple of the num_attention_heads.

        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio, bias=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def swapaxes_for_scores(self, hidden_states):
        """
        Swaps axes and reshapes the input tensor for calculating attention scores in the
        SegformerEfficientSelfAttention class.

        Args:
            self (SegformerEfficientSelfAttention): An instance of the SegformerEfficientSelfAttention class.
            hidden_states (torch.Tensor): A tensor representing the hidden states. It should have a shape of
                (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: A tensor representing the reshaped hidden states. The shape of the tensor will be
                (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.
        """
        new_shape = hidden_states.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        """
        Constructs the self-attention mechanism for the SegformerEfficientSelfAttention class.

        Args:
            self (object): The instance of the SegformerEfficientSelfAttention class.
            hidden_states (tensor): The input tensor representing the hidden states.
                Shape (batch_size, seq_len, num_channels).
            height (int): The height of the input tensor.
            width (int): The width of the input tensor.
            output_attentions (bool, optional): Flag indicating whether to output attentions. Defaults to False.

        Returns:
            tuple: A tuple containing the context layer tensor and attention probabilities tensor if output_attentions
                is True, otherwise only the context layer tensor.

        Raises:
            ValueError: If the shape of the hidden_states tensor is not compatible.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If an error occurs during the computation.
        """
        query_layer = self.swapaxes_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.swapaxes_for_scores(self.key(hidden_states))
        value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SegformerSelfOutput(nn.Module):

    """
    This class represents the self-output of a segmenter transformer model (Segformer) in a neural network architecture.
    It inherits from the nn.Module class.

    Attributes:
        dense (nn.Linear): A fully connected layer that applies linear transformation to the input hidden states.
        dropout (nn.Dropout): A dropout layer that randomly zeros some of the elements of the input tensor.

    Methods:
        __init__: Initializes an instance of the SegformerSelfOutput class.
        forward: Constructs the self-output of the Segformer model.

    """
    def __init__(self, config, hidden_size):
        """
        Initializes an instance of the SegformerSelfOutput class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing various settings.
            hidden_size (int): The size of the hidden layer.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is called when a new instance of the SegformerSelfOutput class is created.
            It initializes the instance by setting up the necessary components for self-attention and output computation.

            The 'config' parameter is an object that stores various settings and configurations for the model.
            It is used to access the hidden dropout probability, which is used in the dropout layer. The 'hidden_size'
            parameter specifies the size of the hidden layer in the model.

            Inside the method, the 'super().__init__()' statement calls the __init__() method of the parent class to ensure
            proper initialization.

            The 'self.dense' attribute is an instance of the nn.Linear class, which represents a fully connected layer.
            It takes the 'hidden_size' as both the input and output size. This layer is used for self-attention computation.

            The 'self.dropout' attribute is an instance of the nn.Dropout class. It takes the 'config.hidden_dropout_prob'
            as the dropout probability. This layer is used for regularization during training to prevent overfitting.

            Note that this method does not perform any computations and is solely responsible for setting up the necessary
            components for the SegformerSelfOutput class.
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Constructs the output of the SegformerSelfOutput class.

        Args:
            self (SegformerSelfOutput): An instance of the SegformerSelfOutput class.
            hidden_states (tensor): The hidden states of the self-attention mechanism.
                These states are passed through a dense layer and a dropout layer.
            input_tensor (tensor): The input tensor to the self-attention mechanism.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):

    """
    This class represents the attention mechanism used in the Segformer model. It inherits from the `nn.Module` class.

    Attributes:
        self (SegformerEfficientSelfAttention): Instance of the SegformerEfficientSelfAttention class that handles
            self-attention computations.
        output (SegformerSelfOutput): Instance of the SegformerSelfOutput class that computes the final attention output.
        pruned_heads (set): A set that stores the indices of pruned attention heads.

    Methods:
        __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
            Initializes the SegformerAttention class.

            Args:

            - config (object): The configuration object.
            - hidden_size (int): The size of the hidden layers.
            - num_attention_heads (int): The number of attention heads.
            - sequence_reduction_ratio (float): The sequence reduction ratio.

        prune_heads(self, heads):
            Prunes the specified attention heads from the model.

            Args:

            - heads (list): A list of attention heads to be pruned.

        forward(self, hidden_states, height, width, output_attentions=False): Constructs the attention mechanism.

            Args:

            - hidden_states (object): The input hidden states.
            - height (int): The height of the input.
            - width (int): The width of the input.
            - output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.

            Returns:

            - tuple: A tuple containing the attention output and any additional outputs.
    """
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        """
        Initializes the SegformerAttention class.

        Args:
            self: The instance of the class.
            config: A configuration object containing various parameters for the attention mechanism.
                - Type: object
                - Purpose: It provides the configuration settings for the attention mechanism.
                - Restrictions: Must be a valid configuration object.
            hidden_size: The size of the hidden layers in the attention mechanism.
                - Type: int
                - Purpose: It defines the dimensionality of the hidden layers.
                - Restrictions: Must be a positive integer.
            num_attention_heads: The number of attention heads to be used in the attention mechanism.
                - Type: int
                - Purpose: It determines the parallel attention computations.
                - Restrictions: Must be a positive integer.
            sequence_reduction_ratio: The ratio by which the input sequence length is reduced in the attention mechanism.
                - Type: int
                - Purpose: It controls the reduction of the input sequence length.
                - Restrictions: Must be a positive integer.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined in the class 'SegformerAttention' and is used to prune the attention heads
        and corresponding linear layers based on the provided 'heads' input.

        Args:
            self (object): The instance of the 'SegformerAttention' class.
            heads (list): A list containing the indices of attention heads to be pruned.
                The indices should be within the valid range of attention heads for the model.

        Returns:
            None.

        Raises:
            ValueError: If the length of the 'heads' list is 0, indicating no heads to be pruned.
            TypeError: If the 'heads' parameter is not provided as a list.
            IndexError: If the indices in the 'heads' list are out of range for the attention heads in the model.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        """
        Construct the attention output of the SegformerAttention module.

        Args:
            self (SegformerAttention): The instance of the SegformerAttention class.
            hidden_states (tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            height (int): The height of the attention output.
            width (int): The width of the attention output.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            tuple: A tuple containing the attention output tensor of shape
                (batch_size, sequence_length, hidden_size),  and any additional outputs as returned by the
                self.self() method.

        Raises:
            None

        """
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SegformerDWConv(nn.Module):

    """
    The SegformerDWConv class represents a depthwise separable convolutional layer for segmentation tasks.
    This class inherits from the nn.Module module.

    Attributes:
        dim (int): The dimensionality of the input and output channels for the depthwise separable convolution.

    Methods:
        __init__: Initializes the SegformerDWConv object with a specified dimensionality for input and output channels.
        forward: Applies the depthwise separable convolution to the input hidden_states and returns the processed output.

    Example:
        ```python
        >>> # Create a SegformerDWConv object with default dimensionality
        >>> seg_dwconv = SegformerDWConv()
        ...
        >>> # Apply the depthwise separable convolution to a set of hidden states
        >>> output = seg_dwconv.forward(hidden_states, height, width)
        ```
    """
    def __init__(self, dim=768):
        """
        Initializes a SegformerDWConv instance.

        Args:
            self: The instance of the SegformerDWConv class.
            dim (int): The dimension of the input and output channels. Defaults to 768.

        Returns:
            None.

        Raises:
            ValueError: If the provided dimension is not a positive integer.
            TypeError: If the provided dimension is not an integer.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, padding=1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        """
        Constructs the SegformerDWConv.

        Args:
            self (SegformerDWConv): An instance of the SegformerDWConv class.
            hidden_states (torch.Tensor): A tensor of shape (batch_size, seq_len, num_channels)
                representing the hidden states.
            height (int): The desired height of the hidden states after transformation.
            width (int): The desired width of the hidden states after transformation.

        Returns:
            None.

        Raises:
            None.
        """
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.swapaxes(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(start_dim=2).swapaxes(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):

    """
    The SegformerMixFFN class represents a feed-forward neural network (FFN) module for the Segformer architecture.
    It is designed to process input features and generate output features using dense layers, depthwise convolution,
    activation functions, and dropout regularization.
    The class inherits from nn.Module and provides methods for initializing the module and forwarding the FFN
    computation graph.

    Attributes:
        config (object): The configuration object containing parameters for the FFN module.
        in_features (int): The number of input features.
        hidden_features (int, optional): The number of hidden features. If not provided, defaults to None.
        out_features (int, optional): The number of output features. If not provided, defaults to None.

    Methods:
        __init__:
            Initializes the SegformerMixFFN module with the provided configuration and feature dimensions.

        forward:
            Constructs the computation graph for the FFN module using the given input hidden_states and spatial
            dimensions (height and width).

    The forwardion of the computation graph involves passing the input through dense layers, depthwise convolution,
    activation functions, and dropout layers to generate the output hidden states.

    Note:
        This docstring is a representation of the class attributes and methods. Please refer to the source code for
        the most accurate and up-to-date information.
    """
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        """
        Initializes an instance of the SegformerMixFFN class.

        Args:
            self: The object itself.
            config (object): The configuration object containing various settings.
            in_features (int): The number of input features.
            hidden_features (int, optional): The number of hidden features. Defaults to None.
            out_features (int, optional): The number of output features. If not provided,
                it will be set equal to in_features.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        '''
        This method forwards the feed-forward network for the SegformerMixFFN class.

        Args:
            self (object): The instance of the SegformerMixFFN class.
            hidden_states (tensor): The input hidden states for the feed-forward network.
            height (int): The height of the input feature map.
            width (int): The width of the input feature map.

        Returns:
            None.

        Raises:
            None
        '''
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""
    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        """
        Initializes a new instance of the SegformerLayer class.

        Args:
            self: The instance of the class.
            config: A configuration object specifying the settings for the SegformerLayer.
            hidden_size (int): The size of the hidden layer.
            num_attention_heads (int): The number of attention heads.
            drop_path (float): The probability of dropping a path during training. Must be between 0.0 and 1.0.
            sequence_reduction_ratio (float): The ratio by which the sequence length is reduced.
            mlp_ratio (float): The ratio by which the hidden size of the Multi-Layer Perceptron (MLP) is computed.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        """
        This method forwards a Segformer layer by performing self-attention and multi-layer perceptron (mlp) operations.

        Args:
            self (object): The instance of the SegformerLayer class.
            hidden_states (tensor): The input tensor representing the hidden states of the layer.
            height (int): The height dimension of the input tensor.
            width (int): The width dimension of the input tensor.
            output_attentions (bool, optional): Flag indicating whether to output attentions. Defaults to False.

        Returns:
            tuple: A tuple containing the output layer and any additional outputs from the layer.

        Raises:
            None.
        """
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class SegformerEncoder(nn.Module):

    """
    SegformerEncoder is a neural network module that represents the encoder of the Segformer model.
    It takes input pixel values and produces a sequence of hidden states that can be used for various downstream tasks.

    Inherits from:
        nn.Module

    Args:
        config: An instance of SegformerConfig that contains various hyperparameters for the encoder.

    Raises:
        ValueError: If the input config is not an instance of SegformerConfig.

    Example:
        ```python
        >>> config = SegformerConfig()
        >>> encoder = SegformerEncoder(config)
        >>> pixel_values = mindspore.Tensor(np.zeros((1, 3, 224, 224)), mindspore.float32)
        >>> outputs = encoder(pixel_values)
        ```
    """
    def __init__(self, config):
        """
        This method initializes a SegformerEncoder instance with the provided configuration.

        Args:
            self (SegformerEncoder): The SegformerEncoder instance.
            config (object): A configuration object containing various parameters for the SegformerEncoder.
                It should include the following attributes:

                - depths (List[int]): The number of layers in each encoder block.
                - drop_path_rate (float): The drop path rate for the network.
                - num_encoder_blocks (int): The number of encoder blocks.
                - patch_sizes (List[int]): The patch sizes for each encoder block.
                - strides (List[int]): The strides for each encoder block.
                - num_channels (int): The number of input channels.
                - hidden_sizes (List[int]): The hidden sizes for each encoder block.
                - num_attention_heads (List[int]): The number of attention heads for each encoder block.
                - sr_ratios (List[float]): The sequence reduction ratios for each encoder block.
                - mlp_ratios (List[float]): The MLP ratios for each encoder block.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration is invalid or incomplete.
            TypeError: If the provided configuration is of an unexpected type.
        """
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [x.item() for x in ops.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Method to forward the SegformerEncoder.

        Args:
            self: The instance of the SegformerEncoder class.
            pixel_values (mindspore.Tensor): The input pixel values as a Tensor.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to False.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to False.
            return_dict (Optional[bool], optional): Whether to return the output as a dictionary. Defaults to True.

        Returns:
            Union[Tuple, BaseModelOutput]:
                The output value which can be either a Tuple or BaseModelOutput.

                - If return_dict is True, it returns a BaseModelOutput containing the last hidden state, hidden states,
                and attentions.
                - If return_dict is False, it returns a Tuple containing the hidden_states, all_hidden_states,
                and all_self_attentions.

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SegformerConfig
    base_model_prefix = "segformer"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


class SegformerModel(SegformerPreTrainedModel):

    """
    A Python class representing a SegformerModel.

    This class is a SegformerModel that inherits from SegformerPreTrainedModel. It is used for performing semantic
    segmentation tasks using the Segformer architecture.

    The SegformerModel class provides methods for initializing the model, pruning model heads, and forwarding the
    model with input pixel values. It also allows for customization of the output, including attention maps and hidden
    states.

    Methods:
        __init__: Initializes the SegformerModel instance with the provided configuration.
        _prune_heads: Prunes specific heads of the model based on the provided dictionary.
        forward: Constructs the model with the given pixel values and returns the output.
            Customization of output options is available.

    Note:
        This class assumes the presence of the SegformerPreTrainedModel class.

    """
    def __init__(self, config):
        """
        Initializes an instance of the SegformerModel class.

        Args:
            self: The instance of the SegformerModel class.
            config (dict):
                A dictionary containing configuration parameters for initializing the SegformerModel.
                It should include the necessary configuration settings for the model.
                Required keys and their datatypes:

                - key1 (datatype): Description.
                - key2 (datatype): Description.
                - ...
                - (Add more keys and descriptions as needed)

        Returns:
            None.

        Raises:
            many exceptions:
                Any exceptions that may be raised during the initialization process should be documented here:

                - ExampleException: Description of the example exception that may be raised.
                - AnotherException: Description of another exception that may be raised.
                 (Add more exceptions and descriptions as needed)
        """
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Constructs the SegformerModel.

        Args:
            self: The instance of the SegformerModel class.
            pixel_values (mindspore.Tensor): The input tensor containing pixel values.
                Shape: (batch_size, num_channels, image_height, image_width).
            output_attentions (Optional[bool], optional): Whether to include attention weights in the output.
                If not provided, it defaults to the value specified in the model's configuration.
                Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to include hidden states in the output.
                If not provided, it defaults to the value specified in the model's configuration.
                Defaults to None.
            return_dict (Optional[bool], optional): Whether to return outputs as a BaseModelOutput dictionary.
                If not provided, it defaults to the value specified in the model's configuration.
                Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutput]: The output of the SegformerModel. If `return_dict` is False,
                it returns a tuple containing the sequence output and the encoder outputs.
                If `return_dict` is True, it returns a BaseModelOutput object with the following attributes:

                - last_hidden_state (mindspore.Tensor): The sequence output of the model.
                Shape: (batch_size, sequence_length, hidden_size).
                - hidden_states (Tuple[mindspore.Tensor]): The hidden states of all layers.
                Each tensor has shape (batch_size, sequence_length, hidden_size).
                - attentions (Tuple[mindspore.Tensor]): The attention weights of all layers.
                Each tensor has shape (batch_size, num_attention_heads, sequence_length, sequence_length).

        Raises:
            None
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SegformerForImageClassification(SegformerPreTrainedModel):

    """
    This class represents a Segformer model for image classification. It is a subclass of SegformerPreTrainedModel.

    The SegformerForImageClassification class initializes and forwards a Segformer model for image classification.
    It takes in a configuration object as a parameter, which includes the number of labels for classification.

    The forwardor (__init__) initializes the SegformerForImageClassification object by calling the superclass's
    forwardor with the provided configuration. It sets the number of labels and creates instances of the
    SegformerModel and nn.Linear classes. The post_init method is then called.

    The forward method forwards the Segformer model for image classification. It takes in several optional
    parameters, including pixel_values (input image tensor), labels (classification labels), output_attentions
    (whether to output attention weights), output_hidden_states (whether to output hidden states), and return_dict
    (whether to return results as a dictionary). It returns a tuple or a SegFormerImageClassifierOutput object.

    The labels parameter is an optional tensor that represents the classification labels for computing the image
    classification/regression loss. The indices in the labels tensor should be in the range of
    [0, ..., config.num_labels - 1]. If config.num_labels == 1, a regression loss (Mean-Square loss) is computed.
    If config.num_labels > 1, a classification loss (Cross-Entropy) is computed.

    The method first calls the SegformerModel's forward method with the provided inputs and optional parameters.
    The output of the forward pass is stored in the sequence_output variable. If the reshape_last_stage configuration
    option is enabled, the sequence_output tensor is permuted and reshaped. Then, the mean of the sequence_output tensor
    is calculated along the second axis.

    The logits tensor is obtained by passing the sequence_output tensor through the classifier module.
    The loss variable is initially set to None.

    If the labels tensor is provided, the problem_type configuration option is checked to determine the type of loss
    calculation. If the problem_type is not set, it is inferred based on the number of labels and the data type of the
    labels tensor. For regression problems with a single label, the problem_type is set to 'regression'.
    For single-label classification problems, the problem_type is set to 'single_label_classification'.
    For multi-label classification problems, the problem_type is set to 'multi_label_classification'.

    The loss is calculated based on the problem_type. For regression problems with a single label, the mean squared error
    (MSE) loss is computed. For single-label classification problems, the cross-entropy loss is computed.
    For multi-label classification problems, the binary cross-entropy with logits loss is computed.

    Finally, the method returns the computed loss and other outputs depending on the value of the return_dict parameter.
    If return_dict is False, the method returns a tuple containing the logits and other outputs.
    If loss is None, the output tuple does not include the loss. If return_dict is True, the method returns a
    SegFormerImageClassifierOutput object containing the loss, logits, hidden states, and attentions.

    Note:
        This docstring does not include the function signatures or any other code.
    """
    def __init__(self, config):
        """
        Initializes a new SegformerForImageClassification instance.

        Args:
            self: The instance of the SegformerForImageClassification class.
            config: An object containing configuration settings for the model.
                It should include the following attributes:

                - num_labels (int): The number of labels for classification.
                - hidden_sizes (list of int): A list of sizes for the hidden layers.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter is missing required attributes.
            RuntimeError: If there is an issue during the initialization process.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.segformer = SegformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # global average pooling
        sequence_output = sequence_output.mean(axis=1)

        logits = self.classifier(sequence_output)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """
    def __init__(self, config: SegformerConfig, input_dim):
        """
        Initializes the SegformerMLP class.

        Args:
            self (object): The instance of the SegformerMLP class.
            config (SegformerConfig): An instance of SegformerConfig containing configuration settings.
            input_dim (int): The dimensionality of the input data.

        Returns:
            None.

        Raises:
            TypeError: If the input arguments are not of the expected types.
            ValueError: If the input_dim is less than or equal to 0.
            RuntimeError: If there is an issue during the initialization process.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: mindspore.Tensor):
        """
        Constructs the SegformerMLP.

        Args:
            self (SegformerMLP): An instance of the SegformerMLP class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = hidden_states.flatten(start_dim=2).swapaxes(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(SegformerPreTrainedModel):
    """
    The `SegformerDecodeHead` class is a subclass of `SegformerPreTrainedModel` and represents the decoding head
    component of the Segformer model.

    This class contains methods for forwarding the decoding head and generating logits for semantic segmentation.

    Attributes:
        linear_c (nn.ModuleList): A list of MLP (Multi-Layer Perceptron) modules for each encoder block.
        linear_fuse (nn.Conv2d): A convolutional layer used for fusing the hidden states of all encoder blocks.
        batch_norm (nn.BatchNorm2d): A batch normalization layer applied to the fused hidden states.
        activation (nn.ReLU): An activation function applied to the hidden states.
        dropout (nn.Dropout): A dropout layer applied to the hidden states.
        classifier (nn.Conv2d): A convolutional layer for generating the final logits.
        config: The configuration object containing hyperparameters and settings for the SegformerDecodeHead.

    Methods:
        forward(encoder_hidden_states: mindspore.Tensor) -> mindspore.Tensor:
            Constructs the decoding head and generates logits for semantic segmentation based on the given encoder
            hidden states.

            Args:

            - encoder_hidden_states (mindspore.Tensor): A tensor containing the hidden states of the encoder blocks.

            Returns:

            - mindspore.Tensor: The logits for semantic segmentation.

    Note:
        - The `SegformerDecodeHead` class requires an instance of `SegformerPreTrainedModel` as its parent class.
        - The decoding head consists of multiple MLP modules, a fusion layer, batch normalization, activation, dropout,
        and a final classifier.
        - The `forward` method takes the encoder hidden states as input and performs the necessary computations to
        generate the logits.
        - The `SegformerDecodeHead` class is designed to be used in conjunction with the Segformer model for semantic
        segmentation tasks.
    """
    def __init__(self, config):
        """
        Initializes the SegformerDecodeHead class.

        Args:
            self: The instance of the SegformerDecodeHead class.
            config: A dictionary containing the configuration parameters for the SegformerDecodeHead,
                including the following keys:

                - num_encoder_blocks (int): The number of encoder blocks.
                - hidden_sizes (list of int): The list of hidden sizes for each encoder block.
                - decoder_hidden_size (int): The size of the hidden layer in the decoder.
                - classifier_dropout_prob (float): The dropout probability for the classifier.
                - num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1, bias=True)

        self.config = config

    def forward(self, encoder_hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        '''
        Constructs the decode head for Segformer.

        Args:
            self (SegformerDecodeHead): The instance of the SegformerDecodeHead class.
            encoder_hidden_states (mindspore.Tensor): The hidden states from the encoder.
                It is a tensor representing the hidden states from the encoder with shape (N, C, H, W).
                N represents the batch size, C represents the number of channels, H represents the height,
                and W represents the width.

        Returns:
            mindspore.Tensor: A tensor representing the logits for the segmentation task with shape (N, C', H', W').
                N represents the batch size, C' represents the number of classes, H' represents the height,
                and W' represents the width.

        Raises:
            ValueError: If the reshape_last_stage configuration is False and the encoder_hidden_state has 3 dimensions.
            RuntimeError: If there is an issue with the linear fusion operation.
            RuntimeError: If there is an issue with the batch normalization operation.
            RuntimeError: If there is an issue with the activation operation.
            RuntimeError: If there is an issue with the dropout operation.
            RuntimeError: If there is an issue with the classifier operation.
        '''
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].shape[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(ops.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class SegformerForSemanticSegmentation(SegformerPreTrainedModel):

    """
    This class represents a Segformer model for semantic segmentation, specifically designed for image processing tasks.
    It is a subclass of SegformerPreTrainedModel.

    The SegformerForSemanticSegmentation class includes methods for model initialization and forwardion.
    It utilizes the SegformerModel and SegformerDecodeHead classes for the main processing steps.

    Methods:
        `__init__`:
            Initializes the SegformerForSemanticSegmentation instance with a given configuration.

            - Parameters:

                - `config`: The configuration object for the model.

        `forward`:
            Constructs the semantic segmentation output based on the input pixel values.

            Parameters:

            - `pixel_values`: A tensor containing the input pixel values for the image.
            - `labels` (Optional): Ground truth semantic segmentation maps for computing the loss.
            Indices should be in the range [0, config.num_labels - 1]. If config.num_labels > 1,
            a classification loss is computed (Cross-Entropy).
            - `output_attentions` (Optional): Boolean flag indicating whether to output attention weights.
            - `output_hidden_states` (Optional): Boolean flag indicating whether to output hidden states.
            - `return_dict` (Optional): Boolean flag indicating whether to return the output as a dictionary.

            Returns:

            - If return_dict is False:

                - If output_hidden_states is True:
                A tuple containing the logits and hidden states (logits, hidden_states).
                - If output_hidden_states is False:
                A tuple containing the logits and attentions (logits, attentions).

            - If return_dict is True:

                - An instance of SemanticSegmenterOutput containing the loss, logits, hidden states
                (if output_hidden_states is True), and attentions.

    Example:
        ```python
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests
        ...
        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ...
        >>> inputs = image_processor(images=image, return_tensors="ms")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```

    """
    def __init__(self, config):
        """
        Initializes an instance of SegformerForSemanticSegmentation.

        Args:
            self: The instance of the SegformerForSemanticSegmentation class.
            config: A dictionary containing configuration parameters for the Segformer model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not a dictionary.
            ValueError: If the config parameter does not contain the required configuration parameters.
        """
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: mindspore.Tensor,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, height, width)`, *optional*):
                Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, SemanticSegmenterOutput]

        Example:
            ```python
            >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            >>> from PIL import Image
            >>> import requests
            ...
            >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = image_processor(images=image, return_tensors="ms")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
            >>> list(logits.shape)
            [1, 150, 128, 128]
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss = F.cross_entropy(upsampled_logits, labels, ignore_index=self.config.semantic_loss_ignore_index)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0).int() & (labels != self.config.semantic_loss_ignore_index).int()).float()
                loss = F.binary_cross_entropy_with_logits(upsampled_logits.squeeze(1), labels.float(), reduction="none")
                loss = (loss * valid_mask).mean()
            else:
                raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

__all__ = [
    "SegformerDecodeHead",
    "SegformerForImageClassification",
    "SegformerForSemanticSegmentation",
    "SegformerLayer",
    "SegformerModel",
    "SegformerPreTrainedModel",
]
