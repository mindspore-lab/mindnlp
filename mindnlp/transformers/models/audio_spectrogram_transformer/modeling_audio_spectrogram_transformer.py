# coding=utf-8
# Copyright 2022 MIT and The HuggingFace Inc. team. All rights reserved.
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
# ============================================================================
""" MindSpore Audio Spectrogram Transformer (AST) model."""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_audio_spectrogram_transformer import ASTConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ASTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "MIT/ast-finetuned-audioset-10-10-0.4593"
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# Audio classification docstring
_SEQ_CLASS_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
_SEQ_CLASS_EXPECTED_OUTPUT = "'Speech'"
_SEQ_CLASS_EXPECTED_LOSS = 0.17


AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    # See all Audio Spectrogram Transformer models at https://hf-mirror.com/models?filter=ast
]


class ASTEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initialize the ASTEmbeddings class.
        
        Args:
            self: The instance of the class.
            config (ASTConfig): An object of type ASTConfig containing configuration settings for AST embeddings.
                It specifies the hidden size and dropout probability for the embeddings.
                
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__()

        self.cls_token = Parameter(ops.zeros(1, 1, config.hidden_size))
        self.distillation_token = Parameter(ops.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ASTPatchEmbeddings(config)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = Parameter(ops.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.config = config

    def get_shape(self, config):
        """
        A method to calculate the output dimensions of the shape based on the provided configuration.
        
        Args:
            self (ASTEmbeddings): The instance of the ASTEmbeddings class.
            config (object):
                The configuration object containing the following attributes:

                - num_mel_bins (int): The number of Mel bins.
                - patch_size (int): The size of the patch.
                - frequency_stride (int): The frequency stride.
                - max_length (int): The maximum length.
                - time_stride (int): The time stride.

        Returns:
            tuple:
                A tuple containing the frequency_out_dimension and time_out_dimension calculated based on the provided configuration.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the configuration attributes do not meet the required constraints for the calculations.
        """
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension

    def construct(self, input_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to construct AST embeddings.

        Args:
            self (ASTEmbeddings): An instance of the ASTEmbeddings class.
            input_values (mindspore.Tensor): The input tensor containing AST token values.
                Shape: (batch_size, sequence_length, hidden_size). Type: float32.

        Returns:
            mindspore.Tensor:
                The constructed AST embeddings tensor.

                - Shape: (batch_size, sequence_length + 2, hidden_size). Type: float32.
                - The tensor includes embeddings for special tokens (CLS token, distillation token),
                - position embeddings, and input token embeddings with positional encodings applied.

        Raises:
            None.
        """
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = ops.cat((cls_tokens, distillation_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ASTPatchEmbeddings(nn.Cell):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ASTRPatchEmbeddings class.

        Args:
            self: An instance of the ASTRPatchEmbeddings class.
            config (object):
                An object containing configuration parameters for the patch embeddings.

                - patch_size (int): The size of the patch in both the frequency and time dimensions.
                - frequency_stride (int): The stride to use in the frequency dimension when extracting patches.
                - time_stride (int): The stride to use in the time dimension when extracting patches.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            1, config.hidden_size,
            kernel_size=(patch_size, patch_size),
            stride=(frequency_stride, time_stride),
            pad_mode='valid',
            has_bias=True
        )

    def construct(self, input_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs patch embeddings for the input tensor.

        Args:
            self (ASTPatchEmbeddings): An instance of the ASTRootEmbeddings class.
            input_values (mindspore.Tensor): The input tensor to construct patch embeddings from. It should have a shape
                of (batch_size, num_patches, embed_dim, embed_dim).

        Returns:
            mindspore.Tensor: A tensor representing the patch embeddings. It has a shape of (batch_size, num_patches,
                embed_dim*embed_dim).

        Raises:
            None.

        """
        input_values = input_values.unsqueeze(1)
        input_values = input_values.swapaxes(2, 3)
        embeddings = self.projection(input_values).flatten(start_dim=2).swapaxes(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->AST
class ASTSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for the AST (Abstract Syntax Tree) model.
    It calculates attention scores and performs attention operations on input hidden states to generate context layers.
    The class includes methods for initializing the self-attention mechanism, reshaping tensors for attention scores calculation,
    and constructing the attention mechanism output.

    The class inherits from nn.Cell and contains the following methods:

    - __init__(self, config: ASTConfig): Initializes the self-attention mechanism with the provided configuration.
    - swapaxes_for_scores(self, x: mindspore.Tensor): Reshapes the input tensor for calculating attention scores.
    - construct(self, hidden_states, head_mask: Optional[mindspore.Tensor] = None, output_attentions: bool = False): Constructs the self-attention mechanism output by performing attention operations on the
    input hidden states.

    The self-attention mechanism involves calculating attention scores, applying softmax to obtain attention probabilities,
    applying dropout, and calculating the context layer.
    Optionally, the method can output attention probabilities along with the context layer.

    Note:
        Ensure that the hidden size is a multiple of the number of attention heads to avoid a ValueError during initialization.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the 'ASTSelfAttention' class.

        Args:
            self: The instance of the class.
            config (ASTConfig): An object of type 'ASTConfig' representing the configuration settings for the self-attention mechanism.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size specified in the 'config' object is not a multiple of the number of attention heads.

        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)
        self.value = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.qkv_bias)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Performs a tensor transformation by swapping axes to prepare the input tensor for self-attention scoring in the ASTSelfAttention class.

        Args:
            self (ASTSelfAttention): The instance of the ASTSelfAttention class.
            x (mindspore.Tensor): The input tensor to be transformed.
                It should have a shape of (batch_size, seq_length, hidden_size).

        Returns:
            mindspore.Tensor: The transformed tensor with swapped axes.
                It has a shape of (batch_size, num_attention_heads, seq_length, attention_head_size).

        Raises:
            None.

        Note:
            - The 'num_attention_heads' and 'attention_head_size' attributes of the ASTSelfAttention class must be set before calling this method.
            - The input tensor should have a shape compatible with the required dimensions for self-attention scoring.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self, hidden_states, head_mask: Optional[mindspore.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
        """
        Constructs the self-attention mechanism in the ASTSelfAttention class.

        Args:
            self (ASTSelfAttention): The instance of the ASTSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states to the self-attention mechanism.
            head_mask (Optional[mindspore.Tensor], optional):
                A tensor used for masking certain heads during attention computation. Defaults to None.
            output_attentions (bool):
                Flag indicating whether to output attention scores alongside context layer. Defaults to False.

        Returns:
            Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
            Returns a tuple containing the context layer tensor and optionally the attention scores tensor.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible for matrix multiplication.
            TypeError: If the input tensors are not of type mindspore.Tensor.
        """
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.swapaxes_for_scores(self.key(hidden_states))
        value_layer = self.swapaxes_for_scores(self.value(hidden_states))
        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->AST
class ASTSelfOutput(nn.Cell):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTSelfOutput class.

        Args:
            self: The instance of the class.
            config (ASTConfig):
                The configuration object containing the hidden size and dropout probability.

                - Type: ASTConfig
                - Purpose: Specifies the configuration for the self output layer.
                - Restrictions: Must be an instance of ASTConfig.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the ASTSelfOutput.

        This method takes three parameters: self, hidden_states, and input_tensor, and returns a mindspore.Tensor object.

        Args:
            self (ASTSelfOutput): An instance of the ASTSelfOutput class.
            hidden_states (mindspore.Tensor): A tensor representing the hidden states.
                This tensor is passed through the dense layer and the dropout layer.
            input_tensor (mindspore.Tensor): A tensor representing the input.

        Returns:
            mindspore.Tensor: A tensor representing the hidden states after passing through the dense and dropout layers.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->AST
class ASTAttention(nn.Cell):

    """
    A class representing an attention mechanism in an Abstract Syntax Tree (AST) model.

    This class implements the attention mechanism used in the AST model.
    It includes methods for pruning attention heads and constructing attention outputs.

    This class inherits from nn.Cell in the MindSpore framework.

    Attributes:
        attention (ASTSelfAttention): The self-attention module used in the AST model.
        output (ASTSelfOutput): The output module used in the AST model.
        pruned_heads (set): A set containing the indices of pruned attention heads.

    Methods:
        __init__:
            Initializes the ASTAttention class with the provided configuration.

        prune_heads:
            Prunes specified attention heads from the model.

        construct:
            Constructs attention outputs based on the provided inputs.

    Returns:
        Tuple: A tuple containing the attention output tensor and additional outputs if requested.

    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTAttention class.

        Args:
            self: The instance of the class.
            config (ASTConfig): An instance of the ASTConfig class representing the configuration settings
            for the attention mechanism. It is required to properly configure the ASTSelfAttention and ASTSelfOutput
            components of the ASTAttention instance.

        Returns:
            None.

        Raises:
            No specific exceptions are raised by this method.
        """
        super().__init__()
        self.attention = ASTSelfAttention(config)
        self.output = ASTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        """
        This method, prune_heads, is a member of the ASTAttention class and is used to prune specific attention heads from the self-attention mechanism.

        Args:
            self: The instance of the ASTAttention class.
            heads (Set[int]): A set of integers representing the indices of attention heads to be pruned.
                Each integer must be a valid index of an attention head in the self-attention mechanism.

        Returns:
            None: This method does not return any value.
                It performs in-place modifications on the self-attention mechanism.

        Raises:
            ValueError: If the input set of attention head indices is empty, a ValueError is raised to indicate an invalid operation.
            IndexError: If any index in the input set of attention heads is out of the valid range for the attention mechanism, an IndexError is raised.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
        """
        Constructs the attention output of the ASTAttention class.

        Args:
            self (ASTAttention): An instance of the ASTAttention class.
            hidden_states (mindspore.Tensor): The input hidden states.
                It should be a tensor of shape (batch_size, sequence_length, hidden_size).
            head_mask (Optional[mindspore.Tensor], optional): The mask indicating the heads to be masked.
                It should be a tensor of shape (num_heads,) or (num_layers, num_heads) with boolean values. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions.
                If True, the attentions are returned as well. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor] or Tuple[mindspore.Tensor]: The attention output.
                If output_attentions is True, it returns a tuple containing the attention output tensor and the attention
            weights tensor. Otherwise, it returns a tuple with only the attention output tensor.

        Raises:
            None

        """
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->AST
class ASTIntermediate(nn.Cell):

    """
    ASTIntermediate represents a neural network cell responsible for intermediate transformations in an Abstract Syntax Tree (AST) model.

    This class inherits from nn.Cell and contains methods for initializing the cell and constructing intermediate transformations on input hidden states.

    Attributes:
        dense (nn.Dense): A dense layer with specified hidden size and intermediate size.
        intermediate_act_fn (function): The activation function used for intermediate transformations.

    Methods:
        __init__: Initializes the ASTIntermediate cell with the given configuration.
        construct: Applies intermediate transformations on the input hidden states and returns the transformed tensor.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTIntermediate class.

        Args:
            self: The instance of the class.
            config (ASTConfig): An object that stores the configuration settings for the ASTIntermediate.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes the ASTIntermediate instance by setting up the necessary attributes and configurations.
            It takes in the following parameters:

            - self: This parameter represents the instance of the class and is automatically passed when calling the method.
            - config (ASTConfig): An object of the ASTConfig class that contains the configuration settings for the ASTIntermediate. It is used to customize the behavior of the instance.

            The method performs the following operations:

            - Calls the __init__ method of the superclass.
            - Initializes the 'dense' attribute of the instance with an instance of the nn.Dense class, using the 'hidden_size' and 'intermediate_size' values from the 'config' object.
            - Checks if the 'hidden_act' attribute of the 'config' object is a string. If it is, it sets the 'intermediate_act_fn' attribute of the instance to the corresponding activation function from the
            ACT2FN dictionary. Otherwise, it sets it to the value of the 'hidden_act' attribute.

            Note: The ACT2FN dictionary is assumed to be defined and accessible within the scope of this class.

        Example:
            ```python
            >>> config = ASTConfig(hidden_size=256, intermediate_size=128, hidden_act='relu')
            >>> intermediate = ASTIntermediate(config)
            ```

            In the example above, a new instance of the ASTIntermediate class is created with the provided 'config' object.
            The 'dense' attribute is initialized with a nn.Dense instance, and the
            'intermediate_act_fn' attribute is set to the 'relu' activation function.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs an intermediate layer of the ASTIntermediate class.

        Args:
            self (ASTIntermediate): An instance of the ASTIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.

        Returns:
            mindspore.Tensor: The output tensor after applying the intermediate layer transformation.

        Raises:
            None.

        This method applies the intermediate layer transformation to the given hidden states tensor.
        First, it passes the hidden states through a dense layer using the 'self.dense' module.
        Then, it applies the intermediate activation function 'self.intermediate_act_fn' to the transformed hidden states.
        The resulting tensor is returned as the output of the intermediate layer.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->AST
class ASTOutput(nn.Cell):

    """
    This class represents the output of an abstract syntax tree (AST) model. It is a subclass of nn.Cell.

    Attributes:
        dense (nn.Dense): A fully connected layer that maps the input tensor to the intermediate size specified in the ASTConfig.
        dropout (nn.Dropout): A dropout layer that applies dropout regularization to the hidden states.

    Methods:
        __init__:
            Initializes the ASTOutput instance.

            Args:
                config (ASTConfig): The configuration object that contains the model's hyperparameters.

        construct:
            Constructs the output of the AST model.

            Args:
                hidden_states (mindspore.Tensor): The hidden states from the previous layer.

                input_tensor (mindspore.Tensor): The input tensor to the ASTOutput layer.

            Returns:
                mindspore.Tensor: The output tensor of the AST model.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTOutput class.

        Args:
            self: The instance of the ASTOutput class.
            config (ASTConfig): An instance of the ASTConfig class containing configuration parameters.
                This parameter specifies the configuration settings for the ASTOutput instance.
                It must be of type ASTConfig.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a new tensor by performing operations on the input hidden_states and input_tensor.

        Args:
            self: The instance of the ASTOutput class.
            hidden_states (mindspore.Tensor):
                A tensor containing the hidden states. It is used as input for the dense layer and dropout operation.
            input_tensor (mindspore.Tensor):
                A tensor containing the input data. It is added to the processed hidden_states.

        Returns:
            mindspore.Tensor:
                A tensor representing the constructed output, which is the result of processing the hidden_states and adding the input_tensor.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->AST
class ASTLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTLayer class.

        Args:
            self: The instance of the ASTLayer class.
            config (ASTConfig): An instance of ASTConfig containing configuration parameters for the layer.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ASTAttention(config)
        self.intermediate = ASTIntermediate(config)
        self.output = ASTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
        """
        Constructs the output of the ASTLayer.

        Args:
            self (ASTLayer): The instance of the ASTLayer.
            hidden_states (mindspore.Tensor): The input hidden states to the layer.
            head_mask (Optional[mindspore.Tensor]): An optional tensor used for masking heads in the self-attention mechanism.
            output_attentions (bool): A flag indicating whether to output attention weights.

        Returns:
            Union[Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor]]:
                A tuple containing the layer output tensor(s). When `output_attentions` is True,
                it returns a tuple with the attention output tensor and the layer output tensor.
                Otherwise, it returns a tuple with only the layer output tensor.

        Raises:
            None.
        """
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->AST
class ASTEncoder(nn.Cell):

    """
    This class represents an AST (Abstract Syntax Tree) Encoder for a neural network model.

    The ASTEncoder class is a subclass of nn.Cell and is responsible for encoding hidden states using multiple layers of ASTLayer.
    It supports optional gradient checkpointing and can output hidden states, self-attentions, or both.

    Attributes:
        config (ASTConfig): The configuration object for the ASTEncoder.
        layer (nn.CellList): A list of ASTLayer instances representing the hidden layers of the encoder.
        gradient_checkpointing (bool): A flag indicating whether gradient checkpointing is enabled.

    Methods:
        __init__:
            Initializes the ASTEncoder instance with the given configuration object.

        construct:
            Constructs the encoder by applying multiple layers of ASTLayer to the input hidden states.
            Optionally, it can apply head masks, output attentions and/or hidden states, and return the results as a tuple or a BaseModelOutput object.

    Returns:
        Union[tuple, BaseModelOutput]: The output of the encoder, which can include hidden states, self-attentions, or both.

    Note:
        - If output_hidden_states is set to True, all_hidden_states will contain all intermediate hidden states.
        - If output_attentions is set to True, all_self_attentions will contain the self-attention scores for each layer.
        - If return_dict is set to False, the output will be returned as a tuple instead of a BaseModelOutput object.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTEncoder class.

        Args:
            self: The instance of the ASTEncoder class.
            config (ASTConfig): An instance of ASTConfig containing configuration parameters for the encoder.
                This parameter is required to initialize the encoder.
                It must be of type ASTConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([ASTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        """
        This method constructs the AST encoder using the provided parameters and returns the hidden states, hidden layers, and self-attentions.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed by the encoder.
            head_mask (Optional[mindspore.Tensor]): Optional mask to be applied to the heads of the encoder.
            output_attentions (bool): Flag indicating whether to output the attentions.
            output_hidden_states (bool): Flag indicating whether to output hidden states.
            return_dict (bool): Flag indicating whether to return the outputs as a dictionary.

        Returns:
            Union[tuple, BaseModelOutput]:
                The return value could be either a tuple containing the hidden states, hidden layers, and self-attentions,
                or a BaseModelOutput object containing the last hidden state, hidden states, and attentions.

        Raises:
            TypeError: If the input types are not as expected.
            IndexError: If the head mask index is out of range.
            RuntimeError: If there is a runtime error during the encoding process.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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


class ASTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ASTConfig
    base_model_prefix = "audio_spectrogram_transformer"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class ASTModel(ASTPreTrainedModel):

    """
    ASTModel is a class representing a model for Abstract Syntax Trees (AST) processing.
    This class inherits from ASTPreTrainedModel and includes methods for initializing the model, getting input embeddings,
    pruning heads, and constructing the model's output.

    Attributes:
        config: An instance of ASTConfig containing configuration parameters for the model.
        embeddings: An instance of ASTEmbeddings for handling AST embeddings.
        encoder: An instance of ASTEncoder for encoding AST inputs.
        layernorm: A layer normalization module with specified hidden size and epsilon.

    Methods:
        __init__: Initializes the ASTModel with the given configuration.
        get_input_embeddings: Returns the patch embeddings used by the model.
        _prune_heads: Prunes specified attention heads in the model.
        construct: Constructs the model output based on input values and optional arguments.

    The construct method handles input processing, encoding, and output generation based on the specified parameters.
    Pruning heads allows for fine-tuning the attention mechanism of the model.
    Overall, ASTModel provides a comprehensive solution for AST-based tasks.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTModel class.

        Args:
            self: The instance of the class.
            config (ASTConfig): The configuration object for the ASTModel.
                t provides necessary settings and hyperparameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        """
        Retrieve the input embeddings from the AST model.

        Args:
            self (ASTModel): The instance of the ASTModel class.
                It represents the current object of the ASTModel.
                This parameter is required as the method is an instance method.

        Returns:
            ASTPatchEmbeddings: An instance of ASTPatchEmbeddings representing the input embeddings.
                The returned ASTPatchEmbeddings object contains the patch embeddings related to the input.

        Raises:
            None
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_values: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        '''
        This method constructs the ASTModel by processing the input values through the model's layers.

        Args:
            self (ASTModel): The instance of the ASTModel.
            input_values (Optional[mindspore.Tensor]): The input values to be processed by the model. Default is None.
            head_mask (Optional[mindspore.Tensor]): The head mask for controlling the attention in the encoder layers. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dict. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: The constructed output, which can be a tuple or BaseModelOutputWithPooling object.

        Raises:
            ValueError: If input_values is None, a ValueError is raised with the message 'You have to specify input_values'.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            raise ValueError("You have to specify input_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ASTMLPHead(nn.Cell):

    """
    ASTMLPHead is a Python class that represents a multi-layer perceptron (MLP) head for an Abstract Syntax Tree (AST) model.
    This class inherits from nn.Cell and contains methods for initializing and constructing the MLP head.

    Attributes:
        layernorm: A LayerNorm module for normalizing the input hidden state.
        dense: A Dense module for applying a linear transformation to the input hidden state.

    Methods:
        __init__: Initializes the ASTMLPHead instance with the provided configuration.
        construct: Constructs the output hidden state by applying layer normalization and linear transformation.

    Note:
        The ASTMLPHead class is designed to be used in conjunction with an AST model for processing abstract syntax trees in natural language processing tasks.
    """
    def __init__(self, config: ASTConfig):
        """
        Initializes an instance of the ASTMLPHead class.

        Args:
            self: The instance of the class.
            config (ASTConfig): The configuration object for the ASTMLPHead.
                It contains the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - num_labels (int): The number of labels for the dense layer. If zero, an identity layer is used instead.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dense = nn.Dense(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def construct(self, hidden_state):
        """
        Constructs the hidden state of the ASTMLPHead.

        Args:
            self (ASTMLPHead): An instance of the ASTMLPHead class.
            hidden_state: The hidden state to be constructed. It should be a tensor-like object.

        Returns:
            None: This method modifies the hidden_state in-place.

        Raises:
            None.
        """
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state


class ASTForAudioClassification(ASTPreTrainedModel):

    """
    ASTForAudioClassification is a class that implements a model for audio classification using the AST (Audio Spectrogram Transformer) architecture.
    This class inherits from ASTPreTrainedModel and provides methods for initializing the model with a configuration, and constructing the model for audio classification tasks.

    Attributes:
        num_labels: Number of labels for the audio classification task.
        audio_spectrogram_transformer: Instance of ASTModel for processing audio input.
        classifier: Instance of ASTMLPHead for classification using the model's pooled output.
    """
    def __init__(self, config: ASTConfig) -> None:
        """
        Initializes an instance of the ASTForAudioClassification class.

        Args:
            self: The instance of the class.
            config (ASTConfig):
                The configuration object containing the necessary parameters for ASTForAudioClassification initialization.

                - num_labels (int): The number of labels/classes for audio classification.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.audio_spectrogram_transformer = ASTModel(config)

        # Classifier head
        self.classifier = ASTMLPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_values: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the audio classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.audio_spectrogram_transformer(
            input_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
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
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
    "ASTForAudioClassification",
    "ASTModel",
    "ASTPreTrainedModel",
]
