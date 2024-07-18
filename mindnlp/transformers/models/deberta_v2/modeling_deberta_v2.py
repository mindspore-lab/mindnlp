# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
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
"""PyTorch DeBERTa-v2 model."""


from collections.abc import Sequence
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_deberta_v2 import DebertaV2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"
_QA_TARGET_START_INDEX = 2
_QA_TARGET_END_INDEX = 9


class ContextPooler(nn.Cell):

    """
    Represents a ContextPooler module used for pooling contextual embeddings in a neural network architecture.
    
    This class inherits from nn.Cell and provides methods for initializing the pooler, constructing the pooled output
    based on hidden states, and retrieving the output dimension.
    The pooler consists of a dense layer and dropout mechanism for processing hidden states.
    
    Attributes:
        dense (nn.Dense): A dense layer for transforming input hidden states to pooler hidden size.
        dropout (StableDropout): A dropout layer for stable dropout operations.
        config: Configuration object containing pooler settings.
    
    Methods:
        __init__: Initializes the ContextPooler with the given configuration.
        construct: Constructs the pooled output by processing hidden states.
        output_dim: Property that returns the output dimension based on the hidden size in the configuration.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ContextPooler class.
        
        Args:
            self: The instance of the ContextPooler class.
            config:
                An object of type 'config' that contains the configuration parameters for the ContextPooler.

                - Type: 'config'
                - Purpose: Specifies the configuration parameters for the ContextPooler.
                - Restrictions: None.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def construct(self, hidden_states):
        """
        Args:
            self (ContextPooler): The instance of the ContextPooler class.
            hidden_states (tensor): A tensor containing hidden states.
                It is expected to have a specific shape and format for processing.
        
        Returns:
            pooled_output (tensor): The output tensor after the pooling operation.
                It represents the pooled context information.
        
        Raises:
            ValueError: If the hidden_states tensor does not meet the expected shape or format requirements.
            RuntimeError: If an error occurs during the pooling operation.
        
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        """
        Method to retrieve the output dimension of the ContextPooler.
        
        Args:
            self (ContextPooler): An instance of the ContextPooler class.
                This parameter is required to access the configuration information.
            
        Returns:
            None: The method does not perform any computation but simply returns the output dimension.
            
        Raises:
            None.
        """
        return self.config.hidden_size


class XSoftmax(nn.Cell):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`mindspore.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:
        ```python
        >>> import torch
        >>> from transformers.models.deberta.modeling_deberta import XSoftmax
        ...
        >>> # Make a tensor
        >>> x = torch.randn([4, 20, 100])
        ...
        >>> # Create a mask
        >>> mask = (x > 0).int()
        ...
        >>> # Specify the dimension to apply softmax
        >>> dim = -1
        ...
        >>> y = XSoftmax.apply(x, mask, dim)
        ```
    """
    def __init__(self, dim=-1):
        """
        Initializes an instance of the XSoftmax class.
        
        Args:
            self: The instance of the XSoftmax class.
            dim (int): The dimension along which the softmax operation is performed. Default is -1.
                The value of dim must be a non-negative integer or -1. If -1, the operation is performed
                along the last dimension of the input tensor.
        
        Returns:
            None.
        
        Raises:
            None.
        
        """
        super().__init__()
        self.dim = dim

    def construct(self, input, mask):
        """
        Constructs a softmax operation with masking for a given input tensor.
        
        Args:
            self (XSoftmax): An instance of the XSoftmax class.
            input (Tensor): The input tensor on which the softmax operation is performed.
            mask (Tensor): A tensor representing the mask used for masking certain elements in the input tensor.
            
        Returns:
            None: The method modifies the input tensor in-place and does not return any value.
        
        Raises:
            TypeError: If the input tensor or the mask tensor is not of the expected type.
            ValueError: If the dimensions of the input tensor and the mask tensor do not match.
            RuntimeError: If an error occurs during the softmax operation or masking process.
        """
        rmask = ~(mask.to(mindspore.bool_))

        output = input.masked_fill(rmask, mindspore.tensor(finfo(input.dtype, 'min')))
        output = ops.softmax(output, self.dim)
        output = output.masked_fill(rmask, 0)
        return output

    def brop(self, input, mask, output, grad_output):
        """
        This method, 'brop', is a member of the 'XSoftmax' class and performs a specific operation on the given input,
        mask, output, and grad_output parameters.
        
        Args:
            self: An instance of the 'XSoftmax' class.
            input: The input parameter of type <input_type>. It represents the input value used in the operation.
            mask: The mask parameter of type <mask_type>. It represents a mask used in the operation.
                <Additional details about the purpose and restrictions of the mask parameter.>
            output: The output parameter of type <output_type>. It represents the output value of the operation.
            grad_output: The grad_output parameter of type <grad_output_type>. It represents the gradient of the output value.
        
        Returns:
            dx: A value of type <dx_type>. It represents the final result of the operation.
                <Additional details about the purpose and format of the dx value.>
            None
        
        Raises:
            <Exception1>: <Description of when and why this exception may be raised.>
            <Exception2>: <Description of when and why this exception may be raised.>
            <Additional exceptions>: <may be raised during the execution of the method.>
        """
        dx = ops.mul(output, ops.sub(grad_output, ops.sum(ops.mul(output, grad_output), self.dim, keepdim=True)))
        return dx, None


class DropoutContext:

    """
    Represents a context for managing dropout operations within a neural network.
    
    This class defines a context for managing dropout operations, including setting the dropout rate, mask,
    scaling factor, and reusing masks across iterations.
    It is designed to be used within a neural network framework to control dropout behavior during training.
    
    Attributes:
        dropout (float): The dropout rate to be applied.
        mask (ndarray or None): The mask array used for applying dropout.
        scale (float): The scaling factor applied to the output.
        reuse_mask (bool): Flag indicating whether to reuse the mask across iterations.
    
    """
    def __init__(self):
        """
        Initialize a DropoutContext object.
        
        Args:
            self: The instance of the DropoutContext class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    """
    Args:
        input (Tensor): The input tensor for which the dropout mask is generated.
        local_context (DropoutContext or float):
            The local context containing information about dropout parameters.

            - If a DropoutContext object is provided, the dropout mask will be generated based on its parameters.
            - If a float value is provided, it will be used as the dropout rate.
    
    Returns:
        None: The function returns the generated dropout mask, or None if no mask is generated.
    
    Raises:
        ValueError: If the local_context is not of type DropoutContext.
    """
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - ops.zeros_like(input).bernoulli(1 - dropout)).to(mindspore.bool_)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(nn.Cell):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""
    def __init__(self, local_ctx):
        """
        Initialize a new instance of the XDropout class.
        
        Args:
            self (object): The instance of the XDropout class.
            local_ctx (object): The local context for the XDropout instance.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.local_ctx = local_ctx
        self.scale = 0
        self.mask = None

    def construct(self, inputs):
        """
        Constructs a masked and scaled version of the input tensor using the XDropout method.
        
        Args:
            self (XDropout): An instance of the XDropout class.
            inputs (torch.Tensor): The input tensor to be masked and scaled.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        mask, dropout = get_mask(inputs, self.local_ctx)
        self.scale = 1.0 / (1 - dropout)
        self.mask = mask
        if dropout > 0:
            return inputs.masked_fill(mask, 0) * self.scale
        return inputs

    # def bprop(self, inputs, outputs, grad_output):
    #     if self.scale > 1:
    #         return grad_output.masked_fill(self.mask, 0) * self.scale
    #     else:
    #         return grad_output


class StableDropout(nn.Cell):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob):
        """Initialize the StableDropout object.
        
        This method is called when a new instance of the StableDropout class is created.
        It initializes the object with the given drop probability and sets the count and context_stack attributes
        to their initial values.
        
        Args:
            self (StableDropout): The instance of the StableDropout class.
            drop_prob (float): The probability of dropping a value during dropout. Must be between 0 and 1 (inclusive).
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def construct(self, x):
        """
        Call the module

        Args:
            x (`mindspore.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout(self.get_context())(x)
        return x

    def clear_context(self):
        """
        Clears the context of the StableDropout class.
        
        Args:
            self (StableDropout): An instance of the StableDropout class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        """
        Initializes the context stack for the StableDropout class.
        
        Args:
            self: The instance of the StableDropout class.
            reuse_mask (bool, optional): Indicates whether the dropout mask should be reused or not. Defaults to True.
            scale (int, optional): The scaling factor applied to the dropout mask. Defaults to 1.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        """
        Args:
            self (StableDropout): The instance of the StableDropout class invoking the method.
                This parameter is required for accessing the instance attributes and methods.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        return self.drop_prob




class DebertaV2SelfOutput(nn.Cell):

    """
    Represents the output layer for the DeBERTa model, responsible for transforming hidden states and
    applying normalization and dropout.
    
    This class inherits from nn.Cell and contains methods to initialize the output layer components,
    including dense transformation, layer normalization, and dropout.
    The 'construct' method takes hidden states and input tensor, applies transformations,
    and returns the final hidden states after normalization and dropout.
    
    Attributes:
        dense (nn.Dense): A fully connected layer for transforming hidden states.
        LayerNorm (DebertaLayerNorm): Layer normalization applied to the hidden states.
        dropout (StableDropout): Dropout regularization to prevent overfitting.
    
    Methods:
        __init__: Initializes the output layer components with the given configuration.
        construct: Applies transformations to hidden states and input tensor to produce final hidden states.
    
    """
    def __init__(self, config):
        """
        Initializes an instance of the DebertaSelfOutput class.
        
        Args:
            self (DebertaSelfOutput): The current instance of the class.
            config: The configuration object containing the settings for the Deberta model.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        Method 'construct' in the class 'DebertaSelfOutput'.
        
        This method constructs the hidden states by applying a series of operations on the input hidden states and the input tensor.
        
        Args:
            self:
                Instance of the DebertaSelfOutput class.

                - Type: DebertaSelfOutput
                - Purpose: Represents the current instance of the class.
            
            hidden_states:
                Hidden states that need to be processed.

                - Type: tensor
                - Purpose: Represents the input hidden states that will undergo transformation.
            
            input_tensor:
                Input tensor to be added to the processed hidden states.

                - Type: tensor
                - Purpose: Represents the input tensor to be added to the processed hidden states.
        
        Returns:
            None.
        
        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Cell):

    """
    This class represents the DebertaAttention module, which is a component of the DeBERTa model.
    It inherits from the nn.Cell class.
    
    DebertaAttention applies self-attention mechanism on the input hidden states, allowing the model to
    focus on different parts of the input sequence. It consists of a DisentangledSelfAttention layer and a
    DebertaSelfOutput layer.
    
    Args:
        config (dict): A dictionary containing the configuration parameters for the DebertaAttention module.
    
    Methods:
        __init__(self, config):
            Initializes a new instance of DebertaAttention.
            
            Args:

            - config (dict): A dictionary containing the configuration parameters for the DebertaAttention module.
                
        construct:

            Applies the DebertaAttention mechanism on the input hidden states.
            
            Args:

            - hidden_states (Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            - attention_mask (Tensor): The attention mask of shape (batch_size, sequence_length, sequence_length)
            where 1 indicates tokens to attend to and 0 indicates tokens to ignore.
            - output_attentions (bool, optional): Whether to output the attention matrix. Defaults to False.
            - query_states (Tensor, optional): The query states of shape (batch_size, sequence_length, hidden_size).
            If not provided, defaults to using the input hidden states.
            - relative_pos (Tensor, optional):
            The relative positions of the tokens of shape (batch_size, sequence_length, sequence_length).
            - rel_embeddings (Tensor, optional):
            The relative embeddings of shape (batch_size, sequence_length, hidden_size).
                
            Returns:
                Tensor or Tuple: The attention output tensor of shape (batch_size, sequence_length, hidden_size)
                or a tuple containing the attention output tensor and the attention matrix if output_attentions
                is True.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the DebertaAttention class.
        
        Args:
            self (DebertaAttention): The current instance of the DebertaAttention class.
            config (object): The configuration object containing the settings for the attention module.
                It should provide the necessary parameters for initializing the DisentangledSelfAttention and
                DebertaSelfOutput instances.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Constructs the DebertaAttention layer with the given parameters.
        
        Args:
            self: The DebertaAttention instance.
            hidden_states (torch.Tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): The attention mask with shape (batch_size, sequence_length).
            output_attentions (bool): Whether to output attention matrices.
            query_states (torch.Tensor): The query states with shape (batch_size, sequence_length, hidden_size).
                If not provided, defaults to hidden_states.
            relative_pos (torch.Tensor):
                The relative position encoding with shape (batch_size, sequence_length, sequence_length).
            rel_embeddings (torch.Tensor):
                The relative position embeddings with shape (num_relative_distances, hidden_size).
        
        Returns:
            None
        
        Raises:
            None
        """
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        return attention_output
# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaV2Intermediate(nn.Cell):

    """
    DebertaIntermediate represents an intermediate layer in the DeBERTa neural network architecture for natural language processing tasks. 
    This class inherits from nn.Cell and contains methods for initializing the layer and performing computations on hidden states. 
    The layer consists of a dense transformation followed by an activation function specified in the configuration. 
    
    Attributes:
        dense (nn.Dense): A dense layer with hidden size and intermediate size specified in the configuration.
        intermediate_act_fn (function): The activation function applied to the hidden states.
    
    Methods:
        __init__(config): Initializes the DebertaIntermediate layer with the provided configuration.
        construct(hidden_states: mindspore.Tensor) -> mindspore.Tensor:
            Applies the dense transformation and activation function to the input hidden states.
    
    """
    def __init__(self, config):
        """
        Initializes a new instance of the DebertaIntermediate class.
        
        Args:
            self: The object itself.
            config (object): An object containing the configuration parameters for the DebertaIntermediate class.
                It should have the following properties:

                - hidden_size (int): The size of the hidden layer in the intermediate module.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or object): The activation function for the hidden layer.

                    - If it is a string, it should be one of the supported activation functions.
                    - If it is an object, it should be a callable that takes a single argument.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the intermediate layer of the Deberta model.
        
        Args:
            self (DebertaIntermediate): The instance of the DebertaIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
        
        Returns:
            mindspore.Tensor: The tensor representing the output hidden states.
        
        Raises:
            None.
        
        This method takes in the hidden states tensor and applies a series of transformations to it in order to
        construct the intermediate layer of the Deberta model. The hidden states tensor is first passed through
        a dense layer, followed by an activation function specified by 'intermediate_act_fn'.
        The resulting tensor represents the intermediate hidden states and is returned as the output of this method.
        
        Note:
            The 'intermediate_act_fn' attribute should be set prior to calling this method to specify the desired
            activation function.
        
        Example:
            ```python
            >>> intermediate_layer = DebertaIntermediate()
            >>> hidden_states = mindspore.Tensor([0.1, 0.2, 0.3])
            >>> output = intermediate_layer.construct(hidden_states)
            ```
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Cell):

    """
    This class represents the output layer of the Deberta model.
    It inherits from the nn.Cell class and is responsible for applying the final transformations to the hidden states.
    
    Attributes:
        dense (nn.Dense): A dense layer that transforms the hidden states to an intermediate size.
        LayerNorm (DebertaLayerNorm): A layer normalization module that normalizes the hidden states.
        dropout (StableDropout): A dropout layer that applies dropout to the hidden states.
        config: The configuration object for the Deberta model.
    
    Methods:
        __init__(self, config):
            Initializes the DebertaOutput instance.
            
            Args:

            - config: The configuration object for the Deberta model.
        
        construct(self, hidden_states, input_tensor):
            Applies the final transformations to the hidden states.
            
            Args:

            - hidden_states: The input hidden states.
            - input_tensor: The original input tensor.
            
            Returns:
                The transformed hidden states after applying the intermediate dense layer, dropout,
                and layer normalization.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the DebertaOutput class.
        
        Args:
            self: The instance of the DebertaOutput class.
            config:
                An instance of the configuration class containing the parameters for the DebertaOutput layer.

                - Type: object
                - Purpose: Specifies the configuration settings for the DebertaOutput layer.
                - Restrictions: Must be a valid instance of the configuration class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def construct(self, hidden_states, input_tensor):
        """
        Constructs the output of the Deberta model by performing a series of operations.
        
        Args:
            self (DebertaOutput): The instance of the DebertaOutput class.
            hidden_states (Tensor): The input hidden states. This tensor represents the intermediate outputs of the model.
            input_tensor (Tensor): The input tensor to be added to the hidden states.
        
        Returns:
            None
        
        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Cell):

    """
    Represents a single layer in the DeBERTa model, containing modules for attention, intermediate processing,
    and output computation.
    
    This class inherits from nn.Cell and is responsible for processing input hidden states through attention mechanisms,
    intermediate processing, and final output computation. It provides a 'construct' method to perform these operations
    and return the final layer output.
    
    Attributes:
        attention (DebertaAttention): Module for performing attention mechanism computation.
        intermediate (DebertaIntermediate): Module for intermediate processing of attention output.
        output (DebertaOutput): Module for computing final output based on intermediate processed data.
    
    Methods:
        construct(hidden_states, attention_mask, query_states=None, relative_pos=None, rel_embeddings=None, output_attentions=False):
            Process the input hidden states through attention, intermediate, and output modules to compute the final layer output.
    
            Args:

            - hidden_states (Tensor): Input hidden states to be processed.
            - attention_mask (Tensor): Mask for attention calculation.
            - query_states (Tensor, optional): Query states for attention mechanism. Default is None.
            - relative_pos (Tensor, optional): Relative position information for attention computation. Default is None.
            - rel_embeddings (Tensor, optional): Relative embeddings for attention computation. Default is None.
            - output_attentions (bool, optional): Flag indicating whether to output attention matrices. Default is False.
    
            Returns:

            - layer_output (Tensor): Final computed output of the layer.
            - att_matrix (Tensor, optional): Attention matrix if 'output_attentions' is True. Otherwise, None.
    
    Note:
        If 'output_attentions' is set to True,
        the 'construct' method will return both the final layer output and the attention matrix.
    """
    def __init__(self, config):
        """
        Initialize a DebertaLayer instance.
        
        Args:
            self (object): The instance of the DebertaLayer class.
            config (object): An object containing configuration settings for the DebertaLayer.
                It is used to customize the behavior of the layer during initialization.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def construct(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        """
        Constructs the DebertaLayer by performing attention, intermediate, and output operations.
        
        Args:
            self (object): The class instance.
            hidden_states (torch.Tensor): The input hidden states tensor.
            attention_mask (torch.Tensor): The attention mask tensor to mask out padded tokens.
            query_states (torch.Tensor, optional): The tensor representing query states for attention computation. 
                Defaults to None.
            relative_pos (torch.Tensor, optional): The tensor representing relative positions for attention computation. 
                Defaults to None.
            rel_embeddings (torch.Tensor, optional): The tensor containing relative embeddings for attention computation. 
                Defaults to None.
            output_attentions (bool): Flag indicating whether to output attention matrices. Defaults to False.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the dimensions of the input tensors are incompatible.
            TypeError: If the input parameters are not of the expected types.
        """
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        return layer_output



class ConvLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, group=groups,pad_mode= 'pad'
        )
        self.LayerNorm = nn.LayerNorm([config.hidden_size],  epsilon=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def construct(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        rmask = (1 - input_mask).bool()
        out.masked_fill(rmask.unsqueeze(-1).broadcast_to(out.shape), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(dtype=layer_norm_input.dtype)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(dtype=output.dtype)
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Cell):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.CellList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.shape[-2] if query_states is not None else hidden_states.shape[-2]
            relative_pos = build_relative_position(
                q,
                hidden_states.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = ops.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = ops.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mindspore.tensor(mid - 1).to(dtype=relative_pos.dtype),
        ops.abs(relative_pos),
    )
    log_pos = (
        ops.ceil(ops.log((abs_pos.to(dtype=mindspore.float32) / mid)) / ops.log(mindspore.tensor((max_position - 1)/ mid)) * (mid - 1)) + mid
    ).to(mindspore.int64)
    bucket_pos = ops.where(abs_pos <= mid, relative_pos.to(dtype=log_pos.dtype), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
        device (`torch.device`): the device on which tensors will be created.

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """

    q_ids = ops.arange(0, query_size)
    k_ids = ops.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(dtype=mindspore.int64)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

@mindspore.jit
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.broadcast_to((query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]))

@mindspore.jit
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.broadcast_to((query_layer.shape[0], query_layer.shape[1], key_layer.shape[-2], key_layer.shape[-2]))

@mindspore.jit
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.broadcast_to(p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))


class DisentangledSelfAttention(nn.Cell):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Dense(config.hidden_size, self.all_head_size, has_bias=True)
        self.key_proj =  nn.Dense(config.hidden_size, self.all_head_size, has_bias=True)
        self.value_proj =  nn.Dense(config.hidden_size, self.all_head_size, has_bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Dense(config.hidden_size, self.all_head_size, has_bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)
        self.softmax = XSoftmax(-1)
    def swapaxes_for_scores(self, x, attention_heads):
        new_x_shape = x.shape[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).view(-1, x.shape[1], x.shape[-1])

    def construct(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.BoolTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.swapaxes_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.swapaxes_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.swapaxes_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = ops.sqrt(mindspore.tensor(query_layer.shape[-1], dtype=mindspore.float32) * scale_factor)
        attention_scores = ops.bmm(query_layer, key_layer.swapaxes(-1, -2) / scale.to(dtype=query_layer.dtype))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.shape[-2], attention_scores.shape[-1]
        )

        # bsz x height x length x dimension
        attention_probs = self.softmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs)
        context_layer = ops.bmm(
            attention_probs.view(-1, attention_probs.shape[-2], attention_probs.shape[-1]), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1])
            .permute(0, 2, 1, 3)
        )
        new_context_layer_shape = context_layer.shape[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(
                q,
                key_layer.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.to(dtype=mindspore.int64)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.swapaxes_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)
            pos_key_layer = self.swapaxes_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
                query_layer.shape[0] // self.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.swapaxes_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.swapaxes_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.shape[0] // self.num_attention_heads, 1, 1)  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = ops.sqrt(mindspore.tensor(pos_key_layer.shape[-1], dtype=mindspore.float32) * scale_factor)
            c2p_att = ops.bmm(query_layer, pos_key_layer.swapaxes(-1, -2))
            c2p_pos = ops.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = ops.gather_elements(
                c2p_att,
                dim=-1,
                index=c2p_pos.squeeze(0).broadcast_to((query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1])),
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = ops.sqrt(mindspore.tensor(pos_query_layer.shape[-1], dtype=mindspore.float32) * scale_factor)
            if key_layer.shape[-2] != query_layer.shape[-2]:
                r_pos = build_relative_position(
                    key_layer.shape[-2],
                    key_layer.shape[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = ops.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = ops.bmm(key_layer, pos_query_layer.swapaxes(-1, -2))
            p2c_att = ops.gather_elements(
                p2c_att,
                dim=-1,
                index=p2c_pos.squeeze(0).broadcast_to((query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2])),
            ).swapaxes(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


# Copied from transformers.models.deberta.modeling_deberta.DebertaEmbeddings with DebertaLayerNorm->LayerNorm
class DebertaV2Embeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes the DebertaEmbeddings class.
        
        Args:
            self (object): Instance of the DebertaEmbeddings class.
            config (object): 
                An object containing configuration parameters for the Deberta model.
                
                - Type: Custom class object.
                - Purpose: Specifies the model configuration including vocab size, hidden size, max position embeddings, 
                type vocab size, etc.
                - Restrictions: Must be a valid configuration object.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Dense(self.embedding_size, config.hidden_size, has_bias=False)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        """
        Constructs the embeddings for the Deberta model.
        
        Args:
            self (DebertaEmbeddings): An instance of the DebertaEmbeddings class.
            input_ids (Tensor, optional):
                A tensor of shape (batch_size, sequence_length) representing the input token IDs. Default is None.
            token_type_ids (Tensor, optional):
                A tensor of shape (batch_size, sequence_length) representing the token type IDs. Default is None.
            position_ids (Tensor, optional):
                A tensor of shape (batch_size, sequence_length) representing the position IDs. Default is None.
            mask (Tensor, optional):
                A tensor of shape (batch_size, sequence_length) representing the attention mask. Default is None.
            inputs_embeds (Tensor, optional):
                A tensor of shape (batch_size, sequence_length, embedding_size) representing the input embeddings.
                Default is None.
        
        Returns:
            Tensor: A tensor of shape (batch_size, sequence_length, embedding_size) representing the constructed embeddings.
        
        Raises:
            None.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.to(dtype=mindspore.int64))
        else:
            position_embeddings = ops.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.ndim != embeddings.ndim:
                if mask.ndim == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel with Deberta->DebertaV2
class DebertaV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DebertaV2Config
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

# Copied from transformers.models.deberta.modeling_deberta.DebertaModel with Deberta->DebertaV2
class DebertaV2Model(DebertaV2PreTrainedModel):

    """
    DebertaModel class represents a DeBERTa model for natural language processing tasks. 
    This class inherits functionalities from DebertaPreTrainedModel and implements methods for initializing the model,
    getting and setting input embeddings, and constructing the model output.
    
    Attributes:
        embeddings (DebertaEmbeddings): The embeddings module of the DeBERTa model.
        encoder (DebertaEncoder): The encoder module of the DeBERTa model.
        z_steps (int): Number of Z steps used in the model.
        config: Configuration object for the model.
    
    Methods:
        __init__: Initializes the DebertaModel with the provided configuration.
        get_input_embeddings: Retrieves the word embeddings from the input embeddings.
        set_input_embeddings: Sets new word embeddings for the input embeddings.
        _prune_heads: Prunes heads of the model based on the provided dictionary.
        construct: Constructs the model output based on the input parameters.
    
    Raises:
        NotImplementedError: If the prune function is called as it is not implemented in the DeBERTa model.
        ValueError: If both input_ids and inputs_embeds are specified simultaneously,
            or if neither input_ids nor inputs_embeds are provided.
    
    Returns:
        Tuple or BaseModelOutput: Depending on the configuration settings, returns either a tuple or a
            BaseModelOutput object containing the model output.
    
    Note:
        This class is designed for use in natural language processing tasks and leverages the DeBERTa architecture
        for efficient modeling.
    
    """
    def __init__(self, config):
        """
        Initializes a new instance of the DebertaModel class.
        
        Args:
            self: The instance of the class.
            config (object): The configuration object containing the model configuration parameters.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = 0
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the DebertaModel.
        
        Args:
            self (DebertaModel): An instance of the DebertaModel class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set the input embeddings for a DebertaModel instance.
        
        Args:
            self (DebertaModel): The instance of the DebertaModel class.
            new_embeddings (object): New input embeddings to be set for the model. 
                It should be of the appropriate type compatible with the model's word_embeddings attribute.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the new_embeddings parameter is not of the expected type.
            ValueError: If the new_embeddings parameter is invalid or incompatible with the model.
        """
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        This method constructs a DebertaModel based on the provided input parameters.
        
        Args:
            self (object): The instance of the DebertaModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing token indices. Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor to specify which tokens should be attended to. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The tensor specifying the type of each token. Default is None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing position indices of tokens. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                The tensor containing precomputed embeddings for input tokens. Default is None.
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Flag to indicate whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag to indicate whether to return output as a dictionary. Default is None.
        
        Returns:
            Union[Tuple, BaseModelOutput]:
                The output value, which can either be a tuple or a BaseModelOutput object, containing
                the constructed DebertaModel.
        
        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified simultaneously.
            ValueError: Raised if neither input_ids nor inputs_embeds are specified.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaPredictionHeadTransform with Deberta->DebertaV2
class DebertaV2PredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.dense = nn.Dense(config.hidden_size, self.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm([self.embedding_size], epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        """
        This method 'construct' is defined within the class 'DebertaPredictionHeadTransform' and is responsible for
        processing the hidden states.
        
        Args:
            self: An instance of the 'DebertaPredictionHeadTransform' class.
            hidden_states: A tensor representing the hidden states to be processed.
                It is of type 'Tensor' and is expected to contain the information to be transformed.
        
        Returns:
            hidden_states: A tensor containing the transformed hidden states after processing.
                It is of type 'Tensor' and represents the result of the transformation operation.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaLMPredictionHead with Deberta->DebertaV2
class DebertaV2LMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(self.embedding_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        This method constructs the prediction head for DebertaLM model.
        
        Args:
            self (DebertaLMPredictionHead): An instance of the DebertaLMPredictionHead class.
            hidden_states (tensor): The hidden states to be processed for prediction.
        
        Returns:
            None: The processed hidden states after passing through the transformation and decoder layers.
        
        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaV2OnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaV2LMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Dense(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    logits = logits.view(-1).to(labels.dtype)
                    loss = ops.mse_loss(logits, labels.view(-1))
                elif labels.ndim == 1 or labels.shape[-1] == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.to(dtype=mindspore.int64)
                    if label_index.shape[0] > 0:
                        labeled_logits = ops.gather_elements(
                            logits, 0, label_index.broadcast_to((label_index.shape[0], logits.shape[1]))
                        )
                        labels = ops.gather_elements(labels, 0, label_index.view(-1))
                        loss = ops.cross_entropy(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = mindspore.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


# Copied from transformers.models.deberta.modeling_deberta.DebertaForTokenClassification with Deberta->DebertaV2
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):

    def __init__(self, config):
        """
        Initializes a new instance of the DebertaForQuestionAnswering class.
        
        Args:
            self: The instance of the class.
            config: An instance of the configuration class containing the model configuration.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Dense(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
__all__ = [
    "DebertaV2ForMaskedLM",
    "DebertaV2ForQuestionAnswering",
    "DebertaV2ForSequenceClassification",
    "DebertaV2ForTokenClassification",
    "DebertaV2Model",
    "DebertaV2PreTrainedModel",
    "DebertaV2ForMultipleChoice"
]
