# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" MindSpore mT5 model."""

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal, Constant

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.core.nn import CrossEntropyLoss
from mindnlp.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5
class MT5LayerNorm(nn.Module):

    """
    Represents a layer normalization module in the MT5 style with no bias and no subtraction of mean.
    
    This class inherits from nn.Module and provides functionality for layer normalization in the MT5 style. 
    The forwardor initializes the layer normalization module with the specified hidden size and epsilon value. 
    The 'forward' method accepts hidden states as input, calculates the variance, and normalizes the hidden states
    using the calculated variance and epsilon value.
    If the weight data type is float16 or bfloat16, the hidden states are converted to the weight data type before
    returning the weighted normalized hidden states.
    
    Attributes:
        hidden_size (int): The size of the hidden states.
        eps (float): The epsilon value for numerical stability.

    Methods:
        __init__: Constructs a MT5LayerNorm module with the given hidden size and epsilon value.
        forward: Applies layer normalization to the input hidden states and returns the normalized output.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the MT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Method to perform layer normalization on hidden states.

        Args:
            self (MT5LayerNorm): The instance of the MT5LayerNorm class.
            hidden_states (Tensor): The input hidden states to be normalized.

        Returns:
            None: This method does not return any value but updates the hidden states in-place after normalization.

        Raises:
            TypeError: If the input hidden_states are not of type Tensor.
            ValueError: If the variance calculation encounters any issues.
            RuntimeError: If there are runtime issues during the normalization process.
        """
        # MT5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->MT5
class MT5DenseActDense(nn.Module):

    """
    MT5DenseActDense is a neural network module that implements a specific architecture for
    processing hidden states in the MT5 model.
    It consists of two dense layers with an activation function and dropout in between.

    Inherits from nn.Module.

    The __init__ method initializes the MT5DenseActDense module with the provided MT5Config object.
    It sets up the internal components including two dense layers, a dropout layer, and an activation function.

    The forward method processes the input hidden states through the internal components in sequence.
    It applies the first dense layer, activation function, dropout, type conversion if necessary, and the
    second dense layer.
    The final processed hidden states are returned as the output of the module.
    """
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5DenseActDense class.

        Args:
            self: The instance of the class.
            config (MT5Config):
                An object of type MT5Config containing configuration parameters.

                - MT5Config.d_model (int): The model dimension.
                - MT5Config.d_ff (int): The feed-forward dimension.
                - MT5Config.dropout_rate (float): The dropout rate.
                - MT5Config.dense_act_fn (str): The activation function to be used.

        Returns:
            None.

        Raises:
            KeyError: If the specified dense activation function in the config is not found in ACT2FN.
            ValueError: If any of the configuration parameters are missing or invalid.
        """
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        """
        This method forwards the hidden states by applying operations and transformations.

        Args:
            self: The instance of the MT5DenseActDense class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed. It should be a tensor.

        Returns:
            mindspore.Tensor: The processed hidden states after applying the operations and transformations.

        Raises:
            TypeError: If the input hidden_states is not of type mindspore.Tensor.
            ValueError: If the weight dtype of self.wo is not compatible with the dtype of hidden_states.
            RuntimeError: If an unexpected error occurs during the processing of hidden_states.
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->MT5
class MT5DenseGatedActDense(nn.Module):

    """
    This class represents a dense gated activation module for the MT5 model. It inherits from the nn.Module class.

    The MT5DenseGatedActDense class contains methods to initialize and forward the dense gated activation module.

    Methods:
        __init__: Initializes the MT5DenseGatedActDense module with the given configuration.
        forward: Constructs the dense gated activation module using the provided hidden states.

    Attributes:
        wi_0: A dense layer that transforms the input hidden states.
        wi_1: A dense layer that transforms the input hidden states.
        wo: A dense layer that transforms the gated hidden states.
        dropout: A dropout layer to apply dropout to the transformed hidden states.
        act: The activation function to be applied to the transformed hidden states.

    Example:
        ```python
        >>> config = MT5Config(d_model=512, d_ff=2048, dropout_rate=0.1, dense_act_fn='gelu')
        >>> dense_gated_act_dense = MT5DenseGatedActDense(config)
        >>> hidden_states = ...
        >>> output = dense_gated_act_dense.forward(hidden_states)
        ```
    """
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5DenseGatedActDense class.

        Args:
            self: The instance of the class.
            config (MT5Config):
                An object of type MT5Config containing configuration parameters for the model.

                - The 'config' parameter is required and must be of type MT5Config.
                - It is used to configure the dimensions and settings for the dense layers in the model.

        Returns:
            None

        Raises:
            ValueError: If the configuration parameters are not provided or are of incorrect type.
            KeyError: If the activation function specified in the configuration is not supported.
        """
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        """
        This method forwards the hidden states by applying a series of transformations.

        Args:
            self (MT5DenseGatedActDense): The instance of the MT5DenseGatedActDense class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.

        Returns:
            None: This method does not return any value explicitly,
                but it updates the hidden states based on the transformations applied.

        Raises:
            TypeError: If the datatype of the hidden_states is not compatible with the datatype of the weight tensor 'wo'.
        """
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->MT5
class MT5LayerFF(nn.Module):

    """
    MT5LayerFF is a Python class representing a feed-forward layer for the MT5 model.
    It inherits from nn.Module and contains methods for initialization and forward propagation.

    The __init__ method initializes the MT5LayerFF instance with the provided configuration.
    It checks if the configuration includes gated activation and assigns the appropriate DenseReluDense module
    accordingly. Additionally, it sets up layer normalization and dropout.

    The forward method applies layer normalization to the input hidden_states, passes it through the DenseReluDense
    module, applies dropout, and returns the updated hidden_states.

    """
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5LayerFF class.

        Args:
            self: The instance of the MT5LayerFF class.
            config (MT5Config): An instance of the MT5Config class containing configuration settings for the MT5 model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MT5Config.
            ValueError: If the config parameter is missing required attributes.
            RuntimeError: If there is an issue with the initialization process.
        """
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = MT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = MT5DenseActDense(config)

        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, hidden_states):
        """
        Constructs the forward pass of the feed-forward layer in the MT5 model.

        Args:
            self (MT5LayerFF): An instance of the MT5LayerFF class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The output hidden states tensor after applying the feed-forward layer,
                with the same shape as the input tensor.

        Raises:
            None: This method does not raise any exceptions.

        Description:
            This method forwards the forward pass for the feed-forward layer in the MT5 model.
            It takes the input hidden states tensor and applies a series of operations to transform it.
            The steps involved in the forward pass are as follows:

            1. Layer Normalization: The input hidden states tensor is first passed through a layer normalization
            operation using self.layer_norm. This operation normalizes the hidden states, making them more
            robust to variations in scale and distribution.
            2. Feed-Forward Transformation: The normalized hidden states tensor is then passed through a feed-forward
            transformation using self.DenseReluDense. This transformation consists of a linear layer followed by a ReLU
            activation function, followed by another linear layer. This operation helps the model learn complex
            non-linear relationships within the hidden states.
            3. Dropout: The output of the feed-forward transformation is then added to the original hidden states tensor
            after applying dropout. Dropout is a regularization technique that randomly sets a fraction of the hidden
            states to zero during training, which helps prevent overfitting and improves generalization.

            The final output hidden states tensor is returned by this method, which has the same shape as the input tensor.

        Note:
            hidden_states: This method does not modify the input hidden states tensor in-place,
                but instead returns a new tensor.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->MT5
class MT5Attention(nn.Module):

    """
    The `MT5Attention` class is a module that implements the attention mechanism used in the MT5 model.
    It is designed to be used as a building block for the Transformer-based models.

    This class inherits from the `nn.Module` class, which is the base class for all neural network modules in MindSpore.

    The main purpose of this class is to compute the attention weights and output of the attention mechanism.
    It takes in the hidden states, mask, key-value states, position bias, past key-value states, layer head mask,
    query length, use cache flag, and output attentions flag as inputs.

    The class provides the following methods:

    - `__init__`: Initializes the `MT5Attention` instance with the given configuration and relative attention bias flag.
    - `prune_heads`: Prunes the specified attention heads from the model.
    - `_relative_position_bucket`: Translates the relative position to a bucket number for relative attention.
    This method is adapted from Mesh Tensorflow.
    - `compute_bias`: Computes the binned relative position bias for the attention mechanism.
    - `forward`: Constructs the attention mechanism by applying self-attention (if `key_value_states` is None) or
    attention over source sentence (provided by `key_value_states`).

    Please refer to the method docstrings for more detailed information on each method and its parameters.
    """
    def __init__(self, config: MT5Config, has_relative_attention_bias=False):
        """
        Initializes an instance of the MT5Attention class.

        Args:
            self: The instance of the class.
            config (MT5Config): An object containing configuration parameters for the attention mechanism.
                The configuration object must have the following attributes:

                - is_decoder (bool): Indicates if the attention mechanism is used in a decoder.
                - relative_attention_num_buckets (int): Number of buckets for relative attention calculations.
                - relative_attention_max_distance (int): Maximum distance for relative attention calculations.
                - d_model (int): Dimensionality of the model.
                - d_kv (int): Dimensionality of the key and value projections.
                - num_heads (int): Number of attention heads.
                - dropout_rate (float): Dropout rate to apply.
            has_relative_attention_bias (bool, optional): Indicates whether relative attention bias is used.
            Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined in the class 'MT5Attention' and is used to prune specific heads in the
        attention mechanism of a MT5 model.

        Args:
            self (object): The instance of the MT5Attention class.
                It is used to access the attributes and methods within the class.
            heads (list): A list of integers representing the indices of the heads to be pruned.
                The indices should be within the range of existing heads in the attention mechanism.

        Returns:
            None: This method does not return any value. It modifies the attributes of the MT5Attention instance in place.

        Raises:
            None:
                However, potential exceptions may arise if the input 'heads' list contains indices that are out of
                bounds of the existing heads or if any of the helper functions called within this method encounter
                errors.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = ops.arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = ops.arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).swapaxes(1, 2)

        def unshape(states):
            """reshape"""
            return states.swapaxes(1, 2).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.swapaxes(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.float(), dim=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->MT5
class MT5LayerSelfAttention(nn.Module):

    """
    This class represents a self-attention mechanism used in the MT5 (Multilingual Translation) model.
    It is designed to be used as a layer within the MT5 model.

    This class inherits from the nn.Module class, which is a base class for all neural network modules in PyTorch.

    Attributes:
        SelfAttention (MT5Attention): An instance of the MT5Attention class that performs the self-attention computation.
        layer_norm (MT5LayerNorm): An instance of the MT5LayerNorm class that applies layer normalization to the hidden states.
        dropout (nn.Dropout): An instance of the nn.Dropout class that applies dropout regularization to the attention output.

    Methods:
        forward:
            This method applies the self-attention mechanism to the input hidden states, optionally using additional
            inputs such as attention mask, position bias, layer head mask, and past key-value states.

            Args:

            - hidden_states (Tensor): The input hidden states to be processed by the self-attention mechanism.
            - attention_mask (Tensor, optional): An attention mask specifying which positions should be attended
            to and which should be ignored. Defaults to None.
            - position_bias (Tensor, optional): A tensor containing position bias values. Defaults to None.
            - layer_head_mask (Tensor, optional): A tensor containing layer and head mask values. Defaults to None.
            - past_key_value (Tuple[Tensor], optional): A tuple containing past key and value tensors. Defaults to None.
            - use_cache (bool, optional): Whether to use caching for the key-value states. Defaults to False.
            - output_attentions (bool, optional): Whether to output the attention values. Defaults to False.

            Returns:

            - Tuple[Tensor]: A tuple containing the updated hidden states and additional outputs depending on the configuration.

    Note:
        - The self-attention mechanism is applied to the input hidden states after they are layer-normalized.
        - The attention output is added to the input hidden states after applying dropout regularization.
        - The method returns a tuple containing the updated hidden states and additional outputs depending on 
        the configuration.
    """
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration settings for the model.
            has_relative_attention_bias (bool, optional): A flag indicating whether to apply relative attention bias.
                Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.SelfAttention = MT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the self-attention layer of the MT5 model.

        Args:
            self (MT5LayerSelfAttention): The instance of the MT5LayerSelfAttention class.
            hidden_states (Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                The hidden states to be passed through the self-attention layer.
            attention_mask (Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                A mask that indicates which tokens should be attended to and which should not.
                Defaults to None.
            position_bias (Tensor, optional): The position bias tensor of shape
                (batch_size, sequence_length, sequence_length). A bias that is added to the attention scores
                for each token. Defaults to None.
            layer_head_mask (Tensor, optional): The layer head mask tensor of shape (num_heads,) or (num_layers, num_heads).
                A mask that indicates which heads should be masked out.
                Defaults to None.
            past_key_value (Tuple[Tensor], optional): The tuple of past key and value tensors.
                It contains the cached key and value tensors from previous time steps.
                Defaults to None.
            use_cache (bool, optional): Whether to use the cache for the attention outputs of each layer.
                Defaults to False.
            output_attentions (bool, optional): Whether to return the attention scores.
                Defaults to False.

        Returns:
            Tuple[Tensor]:
                The outputs of the self-attention layer.
                The tuple contains:

                - hidden_states (Tensor): The updated hidden states after passing through the self-attention layer.
                It has the same shape as the input tensor.
                - attention_scores (Tensor, optional): The attention scores if 'output_attentions' is set to True.
                It has the shape (batch_size, num_heads, sequence_length, sequence_length).
                - position_bias (Tensor, optional): The updated position bias tensor if 'use_cache' is set to True.
                It has the same shape as the input position bias tensor.

        Raises:
            None.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->MT5
class MT5LayerCrossAttention(nn.Module):

    """
    MT5LayerCrossAttention represents a layer for cross-attention mechanism in the MT5 model.

    This class inherits from nn.Module and includes methods for initializing the layer and forwarding the
    cross-attention mechanism.

    Attributes:
        EncDecAttention: An instance of the MT5Attention class for encoder-decoder attention mechanism.
        layer_norm: An instance of the MT5LayerNorm class for layer normalization.
        dropout: An instance of the nn.Dropout class for applying dropout.

    Methods:
        __init__: Initializes the MT5LayerCrossAttention instance with the given configuration.
        forward: Constructs the cross-attention mechanism using the given parameters and returns the outputs.

    """
    def __init__(self, config):
        """
        Initializes an instance of the MT5LayerCrossAttention class.

        Args:
            self (MT5LayerCrossAttention): The instance of the class.
            config (dict): The configuration dictionary containing the settings for the cross-attention layer.

        Returns:
            None.

        Raises:
            ValueError: If the configuration dictionary 'config' is missing required keys or has invalid values.
            TypeError: If the data types of the input parameters are incorrect.
        """
        super().__init__()
        self.EncDecAttention = MT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        """
        This method forwards the cross-attention mechanism in the MT5 model.

        Args:
            self (MT5LayerCrossAttention): The instance of the MT5LayerCrossAttention class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.
            key_value_states (mindspore.Tensor): The key-value states used in attention computation.
            attention_mask (mindspore.Tensor, optional): Mask to avoid attending to specific positions. Default is None.
            position_bias (mindspore.Tensor, optional): Bias values added to the attention scores. Default is None.
            layer_head_mask (mindspore.Tensor, optional): Mask to control which heads are allowed to attend to which positions.
                Default is None.
            past_key_value (tuple, optional): Key and value tensors from the previous time steps. Default is None.
            use_cache (bool, optional): Whether to use cache for faster decoding. Default is False.
            query_length (int, optional): The length of the queries. Default is None.
            output_attentions (bool, optional): Whether to output attention weights. Default is False.

        Returns:
            tuple: A tuple containing the layer's output and additional attention outputs if requested.

        Raises:
            ValueError: If the shape of the input tensors is not compatible.
            TypeError: If the data types of the input parameters are incorrect.
            RuntimeError: If there is an issue during the attention computation process.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->MT5
class MT5Block(nn.Module):

    """
    This class represents a block of the MT5 model, which is a Transformer-based neural network architecture for
    sequence-to-sequence tasks. It consists of a self-attention layer, an optional cross-attention layer, and a
    feed-forward layer.

    Attributes:
        `is_decoder` (bool): Indicates whether the block is used in the decoder part of the model.
        `layer` (nn.ModuleList): A list of layers in the block, including the self-attention, cross-attention,
            and feed-forward layers.

    Methods:
        `forward`: Performs the forward pass of the block, processing the input hidden states and generating the outputs.

    Details:
        The `MT5Block` class inherits from the `nn.Module` class and overrides the `forward` method. The `__init__`
        method initializes the block's attributes, including the `is_decoder` flag and the list of layers.

        The `forward` method takes various input parameters, including the hidden states, attention masks,
        position biases, and layer head masks. It also accepts optional parameters for encoder hidden states and
        attention masks, as well as past key-value states used for caching.

        The method first checks if past key-value states are provided and validates their correctness.
        It then retrieves the self-attention and cross-attention past key-value states from the input if present.

        Next, the method passes the hidden states through the self-attention layer, using the provided attention mask,
        position bias, and layer head mask. The output includes the updated hidden states and the
        present key-value state.

        If the block is a decoder and encoder hidden states are provided, the method performs cross-attention.
        It retrieves the query length and passes the hidden states, encoder hidden states, and other
        parameters to the cross-attention layer. The output includes the updated hidden states and the present key-value state.

        Finally, the method passes the hidden states through the feed-forward layer. It then clamps the hidden states
        to prevent any numerical issues and returns the final hidden states along with any additional outputs, such as
        present key-value states and attention outputs, depending on the value of the `use_cache` parameter.

    Note:
        This class assumes the usage of the MindSpore deep learning framework.

    """
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes a new instance of the MT5Block class.

        Args:
            self: The object itself.
            config (object): The configuration object for MT5Block.
            has_relative_attention_bias (bool, optional): Specifies whether the attention bias is relative or not.
                Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(MT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(MT5LayerCrossAttention(config))

        self.layer.append(MT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the MT5Block.

        This method is responsible for performing the main computations of the MT5Block.
        It takes in multiple parameters and returns None.

        Args:
            self (MT5Block): An instance of the MT5Block class.
            hidden_states (Tensor): The hidden states of the input sequence.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): The attention mask tensor.
                Shape: (batch_size, sequence_length). Default: None.
            position_bias (Tensor, optional): The position bias tensor.
                Shape: (batch_size, sequence_length, sequence_length). Default: None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder sequence.
                Shape: (batch_size, encoder_sequence_length, hidden_size). Default: None.
            encoder_attention_mask (Tensor, optional): The attention mask tensor for the encoder sequence.
                Shape: (batch_size, encoder_sequence_length). Default: None.
            encoder_decoder_position_bias (Tensor, optional): The position bias tensor for encoder-decoder attention.
                Shape: (batch_size, sequence_length, encoder_sequence_length). Default: None.
            layer_head_mask (Tensor, optional): The layer head mask tensor. Shape: (num_layers, num_heads).
                Default: None.
            cross_attn_layer_head_mask (Tensor, optional): The cross-attention layer head mask tensor.
                Shape: (num_layers, num_heads). Default: None.
            past_key_value (Tuple, optional): Tuple containing the past key-value states.
                Shape: (2 or 4, batch_size, num_heads, past_sequence_length, hidden_size). Default: None.
            use_cache (bool, optional): Whether to use caching. Default: False.
            output_attentions (bool, optional): Whether to output attention weights. Default: False.

        Returns:
            None

        Raises:
            ValueError: If the length of past_key_value is not equal to the expected number of past states.
            Warning: If past_key_values is passed to the encoder.
            TypeError: If the data type of hidden_states is not supported.
            TypeError: If the data type of encoder_hidden_states is not supported.
            TypeError: If the data type of hidden_states after cross-attention is not supported.
            TypeError: If the data type of hidden_states after the final layer is not supported.
        """
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16:
            clamp_value = ops.where(
                ops.isinf(hidden_states).any(),
                np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max - 1000,
                np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max,
            )
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == mindspore.float16:
                clamp_value = ops.where(
                    ops.isinf(hidden_states).any(),
                    np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max - 1000,
                    np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max,
                )
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16:
            clamp_value = ops.where(
                ops.isinf(hidden_states).any(),
                np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max - 1000,
                np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max,
            )
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


# Copied from transformers.models.t5.modeling_t5.T5ClassificationHead with T5->MT5
class MT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config: MT5Config):
        """
        Initializes the MT5ClassificationHead class with the provided configuration.

        Args:
            self (MT5ClassificationHead): The instance of the MT5ClassificationHead class.
            config (MT5Config):
                An object containing configuration parameters for the MT5 model.

                - config.d_model (int): The dimension of the model.
                - config.classifier_dropout (float): The dropout rate for the classifier.
                - config.num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MT5Config.
            ValueError: If any of the configuration parameters are missing or invalid.
        """
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the classification head for an MT5 model.

        Args:
            self: Instance of the MT5ClassificationHead class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed by the classification head.

        Returns:
            mindspore.Tensor: The output tensor after processing through the classification head.

        Raises:
            None
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel with T5->MT5, t5->mt5
class MT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MT5Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    _no_split_modules = ["MT5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        """
        This method generates dummy inputs for the MT5PreTrainedModel class.

        Args:
            self: An instance of the MT5PreTrainedModel class.

        Returns:
            None

        Raises:
            None
        """
        input_ids = mindspore.Tensor(DUMMY_INPUTS)
        input_mask = mindspore.Tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(cell, MT5LayerNorm):
            cell.weight.set_data(initializer(Constant(factor * 1.0), cell.weight.shape, cell.weight.dtype))
        elif isinstance(
            cell,
            (MT5Model, MT5ForConditionalGeneration, MT5EncoderModel, MT5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            cell.shared.weight.set_data(initializer(Normal(factor * 1.0),
                                                cell.shared.weight.shape, cell.shared.weight.dtype))
            if hasattr(cell, "lm_head") and not self.config.tie_word_embeddings:
                cell.lm_head.weight.set_data(initializer(Normal(factor * 1.0), cell.lm_head.weight.shape, cell.lm_head.weight.dtype))
            if hasattr(cell, "qa_outputs"):
                cell.qa_outputs.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                            cell.qa_outputs.weight.shape, cell.qa_outputs.weight.dtype))
                cell.qa_outputs.bias.set_data(initializer('zeros', cell.qa_outputs.bias.shape, cell.qa_outputs.bias.dtype))

        elif isinstance(cell, MT5ClassificationHead):
            cell.dense.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.dense.weight.shape, cell.dense.weight.dtype))

            if hasattr(cell.dense, "bias") and cell.dense.bias is not None:
                cell.dense.bias.set_data(initializer('zeros', cell.dense.bias.shape, cell.dense.bias.dtype))
            cell.out_proj.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.out_proj.weight.shape, cell.out_proj.weight.dtype))

            if hasattr(cell.out_proj, "bias") and cell.out_proj.bias is not None:
                cell.out_proj.bias.set_data(initializer('zeros', cell.out_proj.bias.shape, cell.out_proj.bias.dtype))

        elif isinstance(cell, MT5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            cell.wi.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi.weight.shape, cell.wi.weight.dtype))
            if hasattr(cell.wi, "bias") and cell.wi.bias is not None:
                cell.wi.bias.set_data(initializer('zeros', cell.wi.bias.shape, cell.wi.bias.dtype))

            cell.wo.weight.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))

            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.set_data(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))
        elif isinstance(cell, MT5DenseGatedActDense):
            cell.wi_0.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi_0.weight.shape, cell.wi_0.weight.dtype))
            if hasattr(cell.wi_0, "bias") and cell.wi_0.bias is not None:
                cell.wi_0.bias.set_data(initializer('zeros', cell.wi_0.bias.shape, cell.wi_0.bias.dtype))

            cell.wi_1.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi_1.weight.shape, cell.wi_1.weight.dtype))
            if hasattr(cell.wi_1, "bias") and cell.wi_1.bias is not None:
                cell.wi_1.bias.set_data(initializer('zeros', cell.wi_1.bias.shape, cell.wi_1.bias.dtype))

            cell.wo.weight.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))

            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.set_data(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))
        elif isinstance(cell, MT5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            cell.q.weight.set_data(initializer(Normal(factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                                cell.q.weight.shape, cell.q.weight.dtype))
            cell.k.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                cell.k.weight.shape, cell.k.weight.dtype))
            cell.v.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                cell.v.weight.shape, cell.v.weight.dtype))
            cell.o.weight.set_data(initializer(Normal(factor * ((n_heads * key_value_proj_dim) ** -0.5)),
                                                cell.o.weight.shape, cell.o.weight.dtype))
            if cell.has_relative_attention_bias:
                cell.relative_attention_bias.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                    cell.relative_attention_bias.weight.shape, cell.relative_attention_bias.weight.dtype))

    def _shift_right(self, input_ids):
        """
        This method, _shift_right, is a member of the MT5PreTrainedModel class. It shifts the input_ids to the right
        and adds a decoder start token at the beginning.

        Args:
            self (MT5PreTrainedModel): The instance of the MT5PreTrainedModel class.
            input_ids (mindspore.Tensor): The input tensor containing the tokenized input sequence.
                It represents the input sequence to be shifted to the right.

        Returns:
            None.

        Raises:
            ValueError: It may raise a ValueError if either decoder_start_token_id or pad_token_id is not defined in
                the model configuration. The error message provides guidance on how to resolve the issue.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id. "
                "See MT5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


# Copied from transformers.models.t5.modeling_t5.T5Stack with T5->MT5
class MT5Stack(MT5PreTrainedModel):

    """
    The `MT5Stack` class represents a stack of MT5 blocks in the MT5 model. It is a subclass of `MT5PreTrainedModel`
    and is used for both encoding and decoding tasks.

    Attributes:
        `config`: The configuration of the model.
        `embed_tokens`: The token embeddings for the model.
        `is_decoder`: A boolean indicating whether the model is used as a decoder.
        `block`: A list of `MT5Block` instances representing the stack of MT5 blocks.
        `final_layer_norm`: An instance of `MT5LayerNorm` for layer normalization.
        `dropout`: An instance of `nn.Dropout` for dropout regularization.

    Methods:
        `__init__`: Initializes the `MT5Stack` instance.
        `get_input_embeddings`: Returns the token embeddings of the model.
        `set_input_embeddings`: Sets new token embeddings for the model.
        `forward`: Constructs the model by performing the forward pass.

    Note:
        - The `forward` method is the main method of the class that performs the forward pass through
        the stack of MT5 blocks.

    Example:
        ```python
        >>> config = MT5Config(...)
        >>> embed_tokens = nn.Embedding(...)
        >>> stack = MT5Stack(config, embed_tokens)
        >>> input_ids = torch.tensor([...])
        >>> attention_mask = torch.tensor([...])
        >>> output = stack.forward(input_ids=input_ids, attention_mask=attention_mask)
        ```
    """
    def __init__(self, config, embed_tokens=None):
        """
        Initializes an instance of the MT5Stack class.

        Args:
            self: The instance of the class.
            config (MT5Config): The configuration object for the MT5 model.
            embed_tokens (Optional[mindspore.Tensor]): The embedded tokens for the model. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [MT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            Returns the input embeddings for the MT5Stack model.

        Args:
            self (MT5Stack): The instance of the MT5Stack class.

        Returns:
            None: The method returns None, indicating that it does not have a specific return value.

        Raises:
            None
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the MT5Stack.

        Args:
            self (MT5Stack): An instance of the MT5Stack class.
                Represents the current instance of the MT5Stack class.
            new_embeddings (object):
                New embeddings to be set as input embeddings.

                - Type: Any object.
                - Purpose: Specifies the new embeddings to be used as input embeddings for the MT5Stack.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        This method forwards the MT5 model by processing the input data through multiple transformer layers.

        Args:
            self: Reference to the current instance of the class.
            input_ids (optional): Tensor containing the input token IDs. Default is None.
            attention_mask (optional): Tensor containing the attention mask for the input sequence. Default is None.
            encoder_hidden_states (optional): Tensor containing hidden states from the encoder. Default is None.
            encoder_attention_mask (optional): Tensor containing the attention mask for the encoder hidden states.
                Default is None.
            inputs_embeds (optional): Tensor containing the input embeddings. Default is None.
            head_mask (optional): Tensor containing the head mask for the self-attention mechanism. Default is None.
            cross_attn_head_mask (optional): Tensor containing the head mask for cross-attention mechanism.
                Default is None.
            past_key_values (optional): List of past key-value states. Default is None.
            use_cache (optional): Boolean flag indicating whether to use caching. Default is None.
            output_attentions (optional): Boolean flag indicating whether to output attentions. Default is None.
            output_hidden_states (optional): Boolean flag indicating whether to output hidden states. Default is None.
            return_dict (optional): Boolean flag indicating whether to return a dictionary. Default is None.

        Returns:
            None

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided simultaneously or if neither of them is specified.
            ValueError: If the model is not initialized with valid token embeddings.
            ValueError: If use_cache is set to True and the model is not used as a decoder.
            ValueError: If attention_mask is not provided.
            ValueError: If the model is used as a decoder and encoder_hidden_states is not None but encoder_attention_mask is not provided.
            ValueError: If the head masks have incorrect dimensions.
            ValueError: If past_key_values have incorrect dimensions.
            ValueError: If the output of the method is not in the expected format.
            Others: Other possible exceptions raised by internal method calls.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(
                    encoder_hidden_shape, dtype=mindspore.int64
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = ops.ones(num_layers,
num_heads)`.
"""

class MT5Model(MT5PreTrainedModel):
    r"""

    Example:
        ```python
        >>> from transformers import MT5Model, AutoTokenizer
        ...
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> labels = tokenizer(text_target=summary, return_tensors="pt")
        ...
        >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
        >>> hidden_states = outputs.last_hidden_state
        ```
    """
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5Model.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5Model class.

        Args:
            self: The instance of the MT5Model class.
            config (MT5Config): An object of type MT5Config that holds the configuration parameters for the MT5 model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the MT5Model.

        Args:
            self (MT5Model): The instance of the MT5Model class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """Set the input embeddings for the MT5Model.

        This method sets the shared input embeddings for both the encoder and decoder modules in the MT5Model.

        Args:
            self (MT5Model): An instance of the MT5Model class.
            new_embeddings (mindspore.Tensor): The new input embeddings to be set.

        Returns:
            None.

        Raises:
            None.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    def get_encoder(self):
        """
        Returns the encoder of the MT5Model.

        Args:
            self: An instance of the MT5Model class.

        Returns:
            The encoder of the MT5Model.

        Raises:
            None.
        """
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    def get_decoder(self):
        """
        This method returns the decoder associated with the MT5Model instance.

        Args:
            self: The MT5Model instance itself.

        Returns:
            The decoder associated with the MT5Model instance.

        Raises:
            None.
        """
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.t5.modeling_t5.T5Model.forward with T5->MT5, t5->mt5
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        r"""

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MT5Model
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
            >>> model = MT5Model.from_pretrained("mt5-small")
            ...
            >>> input_ids = tokenizer(
            ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
            ... ).input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            ...
            >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for MT5Model.
            >>> # This is not needed for torch's MT5ForConditionalGeneration as it does this internally using labels arg.
            >>> decoder_input_ids = model._shift_right(decoder_input_ids)
            ...
            >>> # forward pass
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class MT5ForConditionalGeneration(MT5PreTrainedModel):
    r"""
    Example:
        ```python
        >>> from transformers import MT5ForConditionalGeneration, AutoTokenizer
        ...
        >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        ```
    """
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5ForConditionalGeneration class.

        Args:
            self: The object instance.
            config (MT5Config):
                The configuration object containing various parameters for the model.

                - `d_model` (int): The dimensionality of the model.
                - `vocab_size` (int): The size of the vocabulary.
                - `num_decoder_layers` (int): The number of layers in the decoder.
                - `is_decoder` (bool): Indicates whether the instance is a decoder.
                - `use_cache` (bool): Indicates whether to use cache during encoding.
                - `is_encoder_decoder` (bool): Indicates whether the instance is an encoder-decoder.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the MT5 model.

        Args:
            self (MT5ForConditionalGeneration): An instance of the MT5ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """
        Set input embeddings for the MT5 model for conditional generation.

        Args:
            self (MT5ForConditionalGeneration): The instance of the MT5ForConditionalGeneration class.
            new_embeddings (Tensor): New input embeddings to be set for the model.
                Should be a tensor of shape [vocab_size, embedding_size] where:

                - vocab_size: Number of tokens in the vocabulary.
                - embedding_size: Dimension of the token embeddings.

                The new_embeddings should match the token embedding requirements of the model.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings provided is not a tensor.
            ValueError: If the shape of the new_embeddings tensor does not match the expected shape
                [vocab_size, embedding_size].
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the MT5 model.

        Args:
            self (MT5ForConditionalGeneration): The instance of the MT5ForConditionalGeneration class.
            new_embeddings (object): The new output embeddings to be set for the model. It can be of any valid type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MT5 model.

        Args:
            self: An instance of the MT5ForConditionalGeneration class.

        Returns:
            embeddings: The output embeddings of the MT5 model.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_encoder
    def get_encoder(self):
        """
        Retrieve the encoder object used for conditional generation in the MT5ForConditionalGeneration class.

        Args:
            self (MT5ForConditionalGeneration): An instance of the MT5ForConditionalGeneration class.
                This parameter is required for accessing the encoder object associated with the instance.

        Returns:
            encoder: The encoder object that is utilized for conditional text generation.

        Raises:
            None.
        """
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder
    def get_decoder(self):
        """
        Method to retrieve the decoder used in the MT5ForConditionalGeneration class.

        Args:
            self: An instance of the MT5ForConditionalGeneration class.

        Returns:
            decoder: This method returns the decoder associated with the MT5ForConditionalGeneration instance.

        Raises:
            None.
        """
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward with T5->MT5, t5->mt5
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`

        Returns:
            `Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MT5ForConditionalGeneration
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
            >>> model = MT5ForConditionalGeneration.from_pretrained("mt5-small")
            ...
            >>> # training
            >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
            >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            ...
            >>> # inference
            >>> input_ids = tokenizer(
            ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
            ... ).input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
            ```
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        This method prepares inputs for generation in the MT5ForConditionalGeneration class.

        Args:
            self (object): The instance of the class.
            input_ids (Tensor): The input token IDs for the model. Shape: [batch_size, sequence_length].
            past_key_values (tuple, optional): The past key values required for fast autoregressive decoding. Default: None.
            attention_mask (Tensor, optional): The attention mask for the input. Shape: [batch_size, sequence_length].
            head_mask (Tensor, optional): The mask for the multi-head attention layers. Shape: [num_heads, sequence_length].
            decoder_head_mask (Tensor, optional): The mask for the decoder's multi-head attention layers. Shape: [num_heads, sequence_length].
            decoder_attention_mask (Tensor, optional): The attention mask for the decoder. Shape: [batch_size, sequence_length].
            cross_attn_head_mask (Tensor, optional): The mask for the cross-attention layers. Shape: [num_heads, sequence_length].
            use_cache (bool, optional): Whether to use the cache for fast decoding. Default: None.
            encoder_outputs (tuple, optional): The outputs of the encoder. Default: None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for the generation including 'decoder_input_ids', 'past_key_values',
                'encoder_outputs', 'attention_mask', 'head_mask', 'decoder_head_mask', 'decoder_attention_mask', 'cross_attn_head_mask',
                and 'use_cache'.

        Raises:
            None
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels
    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare decoder input IDs from labels for conditional generation.

        Args:
            self (MT5ForConditionalGeneration): An instance of the MT5ForConditionalGeneration class.
            labels (mindspore.Tensor): The labels tensor containing the target sequence to be shifted right.

        Returns:
            None: This method returns None as it directly modifies the input labels tensor.

        Raises:
            None.
        """
        return self._shift_right(labels)

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache for the specified `beam_idx` in the `MT5ForConditionalGeneration` class.

        Args:
            self (MT5ForConditionalGeneration): An instance of the MT5ForConditionalGeneration class.
            past_key_values (Tuple): A tuple containing the past key values for the decoder.
                Each element in the tuple represents the past key values for a specific layer.
                Each layer's past key values is a tuple containing the past key values for each attention head in
                that layer.
            beam_idx (Tensor): The index of the beam to reorder the cache for.

        Returns:
            Tuple: The reordered cache for the specified `beam_idx`. The reordered cache has the same structure as the
                input `past_key_values`, but the values are reordered based on the specified `beam_idx`.

        Raises:
            ValueError: If the shape of the reordered_layer_past_states[0] and layer_past_states[0] mismatch.
            ValueError: If the length of reordered_layer_past_states and layer_past_states mismatch.
        """
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class MT5EncoderModel(MT5PreTrainedModel):
    r"""
    Example:
        ```python
        >>> from transformers import MT5EncoderModel, AutoTokenizer
        ...
        >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
        ```
    """
    model_type = "mt5"
    config_class = MT5Config
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of the MT5EncoderModel class.

        Args:
            self: The instance of the MT5EncoderModel class.
            config (MT5Config): An object of type MT5Config containing configuration parameters for the model.
                The config parameter specifies the configuration settings for the MT5 model.
                It must be an instance of the MT5Config class.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MT5Config.
            ValueError: If the config parameter is missing or if any required configuration settings are not provided.
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel.get_input_embeddings
    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the MT5EncoderModel.

        Args:
            self: An instance of the MT5EncoderModel class.

        Returns:
            The shared input embeddings for the MT5EncoderModel.

        Raises:
            None.
        """
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the MT5EncoderModel.

        Args:
            self (MT5EncoderModel): The instance of the MT5EncoderModel class.
            new_embeddings (object): The new embeddings to be set for the input.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of the correct type.
            ValueError: If there is an issue with setting the input embeddings.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel.get_encoder
    def get_encoder(self):
        """
        Returns the encoder of the MT5EncoderModel.

        Args:
            self: An instance of the MT5EncoderModel class.

        Returns:
            encoder: The method returns the encoder of the MT5EncoderModel.

        Raises:
            None.
        """
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # Copied from transformers.models.t5.modeling_t5.T5EncoderModel.forward with T5->MT5, t5->mt5
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutput]:
        r"""

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MT5EncoderModel
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
            >>> model = MT5EncoderModel.from_pretrained("mt5-small")
            >>> input_ids = tokenizer(
            ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
            ... ).input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class MT5ForSequenceClassification(MT5PreTrainedModel):

    """
    This class represents a sequence classification model based on the MT5 architecture.
    It is designed for fine-tuning the MT5 model on sequence classification tasks.

    The `MT5ForSequenceClassification` class inherits from the `MT5PreTrainedModel` class,
    which provides the basic implementation for loading and saving pre-trained MT5 models.

    To initialize an instance of this class, a `MT5Config` object must be passed as a parameter to the forwardor.

    The `MT5ForSequenceClassification` class has the following attributes:

    - `transformer`: An instance of the `MT5Model` class, which represents the main transformer model.
    - `classification_head`: An instance of the `MT5ClassificationHead` class, which represents the classification head of the model.

    The `forward` method is used to process the input and generate the outputs of the model. It takes several input
    tensors as parameters, such as `input_ids`, `attention_mask`, `decoder_input_ids`, etc. The method returns a tuple
    of outputs, including the predicted logits for classification, and other intermediate outputs if requested.

    If labels are provided, the method also calculates the loss based on the predicted logits and the provided labels.
    The loss calculation depends on the `problem_type` specified in the configuration. The supported problem types are
    regression, single-label classification, and multi-label classification.

    Note:
        The `MT5ForSequenceClassification` class does not currently support passing input embeddings instead of input IDs.

    The `MT5ForSequenceClassification` class is designed to be used with the MT5 model for fine-tuning on sequence
    classification tasks. It provides a convenient interface for processing input sequences and generating predictions.

    Please refer to the documentation of the `MT5PreTrainedModel` class for more details on loading and saving
    pre-trained MT5 models.
    """
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        """
        Initializes an instance of MT5ForSequenceClassification.

        Args:
            self: The instance of the MT5ForSequenceClassification class.
            config (MT5Config): An object of type MT5Config containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MT5Config.
            ValueError: If there are any issues during initialization of the transformer, classification head,
                or post_init method.
        """
        super().__init__(config)
        self.transformer = MT5Model(config)
        self.classification_head = MT5ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.forward
    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, Seq2SeqSequenceClassifierOutput]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(ops.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.config.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class MT5ForQuestionAnswering(MT5PreTrainedModel):

    """
    MT5ForQuestionAnswering is a class that represents a Question Answering model based on the MT5 architecture.
    It is a subclass of MT5PreTrainedModel.

    The class includes the following methods:

    - __init__: Initializes an instance of the class with the given configuration.
    - get_input_embeddings: Returns the shared input embeddings.
    - set_input_embeddings: Sets the shared input embeddings to the provided new embeddings.
    - get_encoder: Returns the encoder module of the model.
    - get_decoder: Returns the decoder module of the model.
    - forward: Constructs the model and returns the outputs.

    The 'forward' method takes various input tensors and returns either a tuple of tensors or an instance of
    Seq2SeqQuestionAnsweringModelOutput.

    Please note that this docstring does not include the method signatures or any other code.
    """
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        """Initialize an instance of the MT5ForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config (MT5Config): The configuration object for the MT5 model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_input_embeddings
    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the MT5 model for question answering.

        Args:
            self: An instance of the MT5ForQuestionAnswering class.

        Returns:
            None: The method returns the shared input embeddings.

        Raises:
            This method does not raise any exceptions.
        """
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for both encoder and decoder in the MT5ForQuestionAnswering model.

        Args:
            self (MT5ForQuestionAnswering): The instance of the MT5ForQuestionAnswering class.
            new_embeddings (Tensor): New embeddings to be set as input for both encoder and decoder.
                Should be a tensor of the same shape as the current input embeddings.

        Returns:
            None.

        Raises:
            ValueError: If the shape of the new embeddings does not match the current input embeddings.
            TypeError: If the new_embeddings parameter is not a tensor.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_encoder
    def get_encoder(self):
        """
        Get the encoder object used in the MT5ForQuestionAnswering class.

        Args:
            self: An instance of MT5ForQuestionAnswering.

        Returns:
            encoder: The method returns the encoder object, which is an instance of a specific encoder used in the
                MT5ForQuestionAnswering class.

        Raises:
            None.

        """
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.get_decoder
    def get_decoder(self):
        """
        Method to retrieve the decoder object.

        Args:
            self: An instance of the MT5ForQuestionAnswering class.

        Returns:
            decoder: The method returns the decoder object associated with the MT5ForQuestionAnswering instance.

        Raises:
            None.
        """
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5ForQuestionAnswering.forward
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqQuestionAnsweringModelOutput]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if start_positions is not None and end_positions is not None:
            use_cache = False

        # Copied from models.bart.modeling_bart.BartModel.forward
        #   different to other models, T5 automatically creates decoder_input_ids from
        #   input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

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

            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + decoder_outputs[1:] + encoder_outputs
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class MT5ForTokenClassification(MT5PreTrainedModel):
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForTokenClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = MT5EncoderModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5ForTokenClassification.forward with T5->MT5
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, outputs[2:-1])
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "MT5EncoderModel",
    "MT5ForConditionalGeneration",
    "MT5ForQuestionAnswering",
    "MT5ForSequenceClassification",
    "MT5ForTokenClassification",
    "MT5Model",
    "MT5PreTrainedModel",
    "MT5Stack",
]
