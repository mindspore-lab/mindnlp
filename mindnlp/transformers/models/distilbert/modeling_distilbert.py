# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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
"""MindSpore DistilBERT model"""


import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...activations import get_activation
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_distilbert import DistilBertConfig


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://hf-mirror.com/models?filter=distilbert
]

# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function takes an attention_mask as input and performs the following operations:
    
    1. Calculates the sum of values in the attention_mask along the last axis,
    treating it as an integer array, and stores the result in the 'seqlens_in_batch' variable.
    2. Flattens the attention_mask and finds the indices of non-zero elements, storing them in the 'indices' variable.
    3. Finds the maximum value in the 'seqlens_in_batch' array and assigns it to the 'max_seqlen_in_batch' variable.
    4. Calculates the cumulative sum of 'seqlens_in_batch' along the first axis, and pads it with a zero at the beginning.
    The resulting array is stored in the 'cu_seqlens' variable.

    Args:
        attention_mask (ndarray): A 2D array representing the attention mask.
        The shape of the array is (batch_size, sequence_length). Each element in the array should be a non-negative integer.

    Returns:
        indices (ndarray): A 1D array containing the indices of non-zero elements in the flattened attention_mask.
        cu_seqlens (ndarray): A 1D array representing the cumulative sum of 'seqlens_in_batch' along the first axis,
            padded with a zero at the beginning.
        max_seqlen_in_batch (int): The maximum value in the 'seqlens_in_batch' array.

    Raises:
        None.
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: mindspore.Tensor):
    """
    Create sinusoidal embeddings for given positions and dimensions.

    Args:
        n_pos (int): The number of positions for which to create embeddings.
        dim (int): The dimension of the embeddings.
        out (mindspore.Tensor): The output tensor to store the sinusoidal embeddings.

    Returns:
        None.

    Raises:
        None
    """
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = mindspore.Tensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = mindspore.Tensor(np.cos(position_enc[:, 1::2]))


class Embeddings(nn.Cell):

    """
    Represents a neural network cell for embedding tokens with position embeddings.

    This class inherits from the nn.Cell module and provides methods for initializing and constructing embeddings
    for token sequences. The embedding process includes the use of word embeddings and optional pre-computed word
    embeddings, along with position embeddings.

    The 'Embeddings' class initializes the word embeddings, position embeddings, LayerNorm, and dropout, and constructs
    the embedded tokens by combining input embeddings with position embeddings.
    Additionally, it provides a method for constructing embedded tokens with the option to pass pre-computed word embeddings.

    The 'construct' method takes input token ids and, if provided, pre-computed word embeddings to return the embedded
    tokens with position embeddings.
    The resulting embedded tokens have the shape (bs, max_seq_length, dim), where 'bs' represents the batch size,
    'max_seq_length' represents the maximum sequence length, and 'dim' represents the dimensionality of the embeddings.
    No token_type embeddings are included in the output.

    Parameters:
        input_ids (mindspore.Tensor):
            The token ids to embed. It should be a mindspore.Tensor with the shape (bs, max_seq_length).
        input_embeds (*optional*, mindspore.Tensor):
            The pre-computed word embeddings. It can only be passed if the input ids are `None`.

    Returns:
        mindspore.Tensor:
            The embedded tokens (plus position embeddings, no token_type embeddings) with the shape (bs, max_seq_length, dim).

    Note:
        The 'Embeddings' class requires the 'config' parameter of type 'PretrainedConfig' during initialization to configure the embeddings.

    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes an instance of the Embeddings class.

        Args:
            self: The instance of the class.
            config (PretrainedConfig):
                The configuration object containing the parameters for the embeddings.

                 - vocab_size (int): The size of the vocabulary.
                 - dim (int): The dimensionality of the embeddings.
                 - pad_token_id (int): The ID of the padding token.
                 - max_position_embeddings (int): The maximum number of positions for the position embeddings.
                 - sinusoidal_pos_embds (bool): Determines whether to use sinusoidal position embeddings.
                 - dropout (float): The dropout rate for the embeddings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.dropout)
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))

    def construct(self, input_ids: mindspore.Tensor, input_embeds: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Parameters:
            input_ids (mindspore.Tensor):
                mindspore.Tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, mindspore.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns:
            mindspore.Tensor(bs, max_seq_length, dim):
                The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.shape[1]

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = ops.arange(seq_length, dtype=mindspore.int64)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Cell):

    """
    This class represents a multi-head self-attention mechanism used in neural networks.
    It is designed to be used as a building block for Transformer-based models.
    The class inherits from the nn.Cell class and implements methods for initializing the attention mechanism,
    pruning heads, and constructing the attention weights and contextualized layer.

    Attributes:
        config (PretrainedConfig): The configuration object containing the parameters for the attention mechanism.
        n_heads (int): The number of attention heads.
        dim (int): The dimensionality of the attention mechanism.
        dropout (nn.Dropout): The dropout layer applied to the attention weights.
        is_causal (bool): Indicates whether the attention mechanism is causal or not.
        q_lin (nn.Dense): Linear layer for query projection.
        k_lin (nn.Dense): Linear layer for key projection.
        v_lin (nn.Dense): Linear layer for value projection.
        out_lin (nn.Dense): Linear layer for the output projection.
        pruned_heads (Set[int]): Set of pruned attention heads.
        attention_head_size (int): The size of each attention head.

    Methods:
        __init__: Initializes the MultiHeadSelfAttention object with the provided configuration.
        prune_heads: Prunes the specified attention heads from the attention mechanism.
        construct:
          Constructs the attention weights and contextualized layer.
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes the MultiHeadSelfAttention class.

        Args:
            self: The instance of the MultiHeadSelfAttention class.
            config (PretrainedConfig): The configuration object used for initializing the MultiHeadSelfAttention.
                It contains parameters such as number of heads (n_heads), dimension (dim), and
                attention dropout (attention_dropout).

        Returns:
            None.

        Raises:
            ValueError: If the number of heads (n_heads) specified in the config does not evenly divide
                the dimension (dim).

        """
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = False

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Dense(config.dim, config.dim)
        self.k_lin = nn.Dense(config.dim, config.dim)
        self.v_lin = nn.Dense(config.dim, config.dim)
        self.out_lin = nn.Dense(config.dim, config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        """
        This method 'prune_heads' is defined within the 'MultiHeadSelfAttention' class and is used to prune specific 
        attention heads in the multi-head self-attention mechanism.

        Args:
            self: The instance of the MultiHeadSelfAttention class.
            heads (List[int]): A list of integers representing the attention heads to be pruned. 
                It identifies the specific attention heads to be removed from the attention mechanism.

        Returns:
            None: This method does not return a value as it operates directly on the instance attributes.

        Raises:
            None.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, axis=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        mask: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, ...]:
        """
        Parameters:
            query: mindspore.Tensor(bs, seq_length, dim)
            key: mindspore.Tensor(bs, seq_length, dim)
            value: mindspore.Tensor(bs, seq_length, dim)
            mask: mindspore.Tensor(bs, seq_length)

        Returns:
            weights: mindspore.Tensor(bs, n_heads, seq_length, seq_length)
            Attention weights context: mindspore.Tensor(bs, seq_length, dim) Contextualized layer.
                Optional: only if `output_attentions=True`
        """
        bs = query.shape[0]
        k_length = key.shape[1]
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.shape == value.shape

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: mindspore.Tensor) -> mindspore.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).swapaxes(1, 2)

        def unshape(x: mindspore.Tensor) -> mindspore.Tensor:
            """group heads"""
            return x.swapaxes(1, 2).view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = ops.matmul(q, k.swapaxes(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, finfo(scores.dtype, 'min')
        )  # (bs, n_heads, q_length, k_length)

        weights = ops.softmax(scores, axis=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = ops.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        return (context,)


class FFN(nn.Cell):

    """A class that represents a Feed-Forward Network (FFN) implemented as a neural network cell.

    The FFN class inherits from the nn.Cell class of the MindSpore framework. It is designed to process input tensors using a
    two-layer fully connected feed-forward network. The class implements the forward pass of the network, applying linear
    transformations, activation functions, and dropout regularization.

    Attributes:
        dropout (mindspore.nn.Dropout): The dropout regularization layer.
        chunk_size_feed_forward (int): The size of chunks used in the forward pass.
        seq_len_dim (int): The dimension corresponding to the sequence length in the input tensor.
        lin1 (mindspore.nn.Dense): The first fully connected layer of the FFN.
        lin2 (mindspore.nn.Dense): The second fully connected layer of the FFN.
        activation (Callable): The activation function used in the network.

    Methods:
        construct:
            Applies the forward pass of the FFN to the input tensor.

        ff_chunk:
            Performs a chunk of the forward pass on the input tensor.

    Note:
        This class assumes that the input tensor has shape (batch_size, sequence_length, dim).

    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes an instance of the FFN (Feed Forward Network) class.

        Args:
            self: The object itself.
            config (PretrainedConfig): The configuration object containing various settings for the FFN.
                The object should have the following attributes:

                 - dropout (float): The dropout probability used in the FFN.
                 - chunk_size_feed_forward (int): The chunk size for feed-forward operations.
                 - dim (int): The input dimension of the FFN.
                 - hidden_dim (int): The hidden dimension of the FFN.
                 - activation (str): The activation function used in the FFN.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Dense(config.dim, config.hidden_dim)
        self.lin2 = nn.Dense(config.hidden_dim, config.dim)
        self.activation = get_activation(config.activation)

    def construct(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """Constructs the feedforward network.

        Args:
            self (FFN): An instance of the FFN class.
            input (mindspore.Tensor): The input tensor of shape [batch_size, sequence_length, hidden_size].

        Returns:
            mindspore.Tensor: The output tensor of shape [batch_size, sequence_length, hidden_size].

        Raises:
            TypeError: If input is not an instance of the mindspore.Tensor class.
        """
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        Performs a forward pass through the feedforward network (FFN) chunk.

        Args:
            self (FFN): The instance of the FFN class.
            input (mindspore.Tensor): The input tensor to the FFN chunk.

        Returns:
            mindspore.Tensor: The output tensor after passing through the FFN chunk.

        Raises:
            None.

        This method applies a series of operations to the input tensor to perform a forward pass through the FFN chunk.
        The operations include linear transformation, activation function, linear transformation, and dropout.
        The resulting tensor is then returned as the output of the FFN chunk.
        """
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


DISTILBERT_ATTENTION_CLASSES = {
    "eager": MultiHeadSelfAttention,
}


class TransformerBlock(nn.Cell):

    """
    The TransformerBlock class represents a single block of the Transformer architecture.
    It is used to process the input data and produce contextualized outputs.

    This class inherits from the nn.Cell class.

    Methods:
        construct:
            Constructs the TransformerBlock using the given input tensors and parameters.

            Parameters:

            - x: mindspore.Tensor(bs, seq_length, dim). The input tensor representing the data to be processed.
            - attn_mask: Optional[mindspore.Tensor]. An optional attention mask tensor of shape (bs, seq_length) to mask out specific positions in the input tensor.
            - head_mask: Optional[mindspore.Tensor]. An optional head mask tensor to mask out specific heads of the attention mechanism.
            - output_attentions: bool. A flag indicating whether to return the attention weights in the output.

            Returns: Tuple[mindspore.Tensor, ...]. A tuple containing the following elements:

            - sa_weights: mindspore.Tensor(bs, n_heads, seq_length, seq_length). The attention weights.
            - ffn_output: mindspore.Tensor(bs, seq_length, dim). The output of the transformer block contextualization.

            Raises:

            - TypeError: If sa_output is not a tuple when output_attentions is True.
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initialize a TransformerBlock instance with the provided configuration.

        Args:
            self (TransformerBlock): The instance of the TransformerBlock class.
            config (PretrainedConfig): The configuration object containing parameters for the TransformerBlock.
                It must be an instance of PretrainedConfig class.
                The 'dim' parameter specifies the dimensionality of the input data.
                The 'n_heads' parameter specifies the number of attention heads to use.
                'n_heads' must be a factor of 'dim' for proper division.

        Returns:
            None.

        Raises:
            ValueError: If the division of 'dim' by 'n_heads' results in a non-zero remainder,
                indicating that 'n_heads' does not evenly divide 'dim'.
        """
        super().__init__()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = DISTILBERT_ATTENTION_CLASSES["eager"](config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, epsilon=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, epsilon=1e-12)

    def construct(
        self,
        x: mindspore.Tensor,
        attn_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, ...]:
        """
        Parameters:
            x: mindspore.Tensor(bs, seq_length, dim)
            attn_mask: mindspore.Tensor(bs, seq_length)

        Returns:
            sa_weights (mindspore.Tensor(bs, n_heads, seq_length, seq_length)): The attention weights
            ffn_output (mindspore.Tensor(bs, seq_length, dim)): The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            if not isinstance(sa_output, tuple):
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: mindspore.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Cell):

    """
    This class represents a Transformer model, which is a type of neural network architecture commonly used in
    natural language processing tasks.

    The Transformer class initializes with a configuration object that contains information about the model's architecture.
    It creates a list of TransformerBlock instances, one for each layer specified in the configuration.
    The number of layers is stored in the 'n_layers' attribute.

    The 'construct' method takes as input an embedded input sequence tensor, 'x', and optionally an attention mask tensor,
    'attn_mask'. It returns the hidden state tensor of the last layer, 'hidden_state', along with additional outputs
    depending on the specified options.
    If 'output_hidden_states' is True, it also returns a tuple containing the hidden states from each layer, 'all_hidden_states'.
    If 'output_attentions' is True, it also returns a tuple containing the attention weights from each layer,
    'all_attentions'.
    The method also supports returning the outputs as a 'BaseModelOutput' object if the 'return_dict' flag is set to True.

    Note:
        The 'construct' method internally iterates through each layer of the Transformer model and applies the layer
        module to the input tensor.
        It accumulates the hidden states and attention weights if the corresponding output options are enabled.

    Parameters:
        x: A tensor of shape (bs, seq_length, dim) representing the embedded input sequence.
        attn_mask: An optional tensor of shape (bs, seq_length) representing the attention mask on the sequence.

    Returns:
        hidden_state: A tensor of shape (bs, seq_length, dim) representing the sequence of hidden states in the last (top) layer.
        all_hidden_states: A tuple of tensors, each of shape (bs, seq_length, dim),
            containing the hidden states from each layer. This is only returned if 'output_hidden_states' is set to True.
        all_attentions: A tuple of tensors, each of shape (bs, n_heads, seq_length, seq_length),
            containing the attention weights from each layer. This is only returned if 'output_attentions' is set to True.

    Raises:
        ValueError: If the length of 'layer_outputs' is not as expected based on the output options.

    Note:
        The 'construct' method builds the Transformer model by sequentially applying the layer modules to the input tensor.
        It uses the 'head_mask' tensor for applying head-wise masking during attention operations.

    Example:
        ```python
        >>> config = PretrainedConfig(n_layers=6)
        >>> transformer = Transformer(config)
        >>> input_tensor = mindspore.Tensor(bs, seq_length, dim)
        >>> output = transformer.construct(input_tensor, attn_mask, head_mask, output_attentions=True, output_hidden_states=True)
        ```
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes an instance of the Transformer class.

        Args:
            self (Transformer): The instance of the Transformer class.
            config (PretrainedConfig): A PretrainedConfig object containing configuration parameters for the Transformer.
                The 'n_layers' attribute of the PretrainedConfig object specifies the number of layers in the Transformer.
                This parameter is required for setting up the Transformer instance.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.CellList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        x: mindspore.Tensor,
        attn_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[mindspore.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: mindspore.Tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: mindspore.Tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: mindspore.Tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[mindspore.Tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[mindspore.Tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                hidden_state,
                attn_mask,
                head_mask[i],
                output_attentions,
            )

            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class DistilBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    load_tf_weights = None
    base_model_prefix = "distilbert"

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
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class DistilBertModel(DistilBertPreTrainedModel):

    """
    Represents a DistilBERT model for natural language processing tasks.

    This class inherits from DistilBertPreTrainedModel and implements various methods for handling position embeddings,
    input embeddings, head pruning, and model construction.
    The class provides functionality to initialize the model, resize position embeddings, get position embeddings,
    resize input embeddings, set input embeddings, prune heads, and construct the model for inference.

    Methods:
        get_position_embeddings: Returns the position embeddings.
        resize_position_embeddings: Resizes position embeddings of the model.
        get_input_embeddings: Returns the input embeddings.
        set_input_embeddings: Sets new input embeddings.
        _prune_heads: Prunes heads of the model based on the provided dictionary.
        construct: Constructs the model for inference with optional parameters.

    Note:
        This class assumes familiarity with the DistilBERT model architecture and its specific components.
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes a new instance of the DistilBertModel class.

        Args:
            self: The instance of the DistilBertModel class.
            config (PretrainedConfig):
                An instance of the PretrainedConfig class containing the configuration settings for the model.
                This parameter is required to configure the model's embeddings and transformer components.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                    old_position_embeddings_weight
                )
            else:
                self.embeddings.position_embeddings.weight = nn.Parameter(
                    old_position_embeddings_weight[:num_position_embeds_diff]
                )

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Retrieve the input embeddings for the DistilBertModel.

        Args:
            self (DistilBertModel): An instance of the DistilBertModel class.
                Represents the current instance of the DistilBertModel.
                This parameter is required for accessing the embeddings.

        Returns:
            nn.Embedding: An instance of the nn.Embedding class.
                Represents the word embeddings used for input to the DistilBertModel.
                These embeddings are used to convert input tokens to dense vectors for processing.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """
        Sets the input embeddings for the DistilBertModel.

        Args:
            self (DistilBertModel): The instance of the DistilBertModel class.
            new_embeddings (nn.Embedding): The new embeddings to be set for the input.

        Returns:
            None.

        Raises:
            None.

        This method allows setting new embeddings for the input in the DistilBertModel.
        The 'self' parameter refers to the instance of the DistilBertModel class on which the method is being called.
        The 'new_embeddings' parameter is of type 'nn.Embedding' and represents the new embeddings to be set for the input.

        Example:
            ```python
            >>> model = DistilBertModel()
            >>> embeddings = nn.Embedding(vocab_size, embedding_dim)
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[mindspore.Tensor, ...]]:
        """
        Constructs a DistilBertModel.

        Args:
            self (DistilBertModel): The instance of the DistilBertModel class.
            input_ids (Optional[mindspore.Tensor]): Input tensor containing the indices of input sequence tokens.
                Default is None.
            attention_mask (Optional[mindspore.Tensor]): Mask tensor indicating which tokens should be attended to.
                Default is None.
            head_mask (Optional[mindspore.Tensor]): Mask tensor indicating which heads should be masked out.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                Input tensor containing the embedded representation of input sequence tokens. Default is None.
            output_attentions (Optional[bool]): Whether to return attention weights. Default is None.
            output_hidden_states (Optional[bool]): Whether to return hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a BaseModelOutput instead of a tuple. Default is None.

        Returns:
            Union[BaseModelOutput, Tuple[mindspore.Tensor, ...]]:
                The output of the DistilBertModel.

                - If `return_dict` is set to True, a BaseModelOutput object is returned.
                - Otherwise, a tuple containing a tensor and optionally, attention weights and hidden states is returned.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are specified simultaneously.
            ValueError: If neither `input_ids` nor `inputs_embeds` are specified.
            Exception: Any other exception that may occur during execution.
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

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)  # (bs, seq_length)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class DistilBertForMaskedLM(DistilBertPreTrainedModel):

    """
    A class representing a DistilBERT model for Masked Language Modeling (MLM).

    This class inherits from the DistilBertPreTrainedModel class and includes methods for initializing the model,
    resizing position embeddings, getting and setting output embeddings, and constructing the model.

    Attributes:
        activation (function): Activation function used in the model.
        distilbert (DistilBertModel): DistilBERT model.
        vocab_transform (nn.Dense): Dense layer for transforming the vocabulary.
        vocab_layer_norm (nn.LayerNorm): Layer normalization for the vocabulary.
        vocab_projector (nn.Dense): Dense layer for projecting the vocabulary.
        mlm_loss_fct (nn.CrossEntropyLoss): Cross-entropy loss function for MLM.

    Methods:
        __init__: Initializes the DistilBertForMaskedLM model.
        get_position_embeddings: Returns the position embeddings.
        resize_position_embeddings: Resizes position embeddings of the model.
        get_output_embeddings: Returns the output embeddings.
        set_output_embeddings: Sets the output embeddings.
        construct: Constructs the DistilBertForMaskedLM model.

    Please see the documentation for the DistilBertPreTrainedModel class for more information on inherited attributes
    and methods.
    """
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        """
        Initializes a new instance of DistilBertForMaskedLM.

        Args:
            self: The object itself.
            config (PretrainedConfig): The configuration for the pretrained model.
                It contains the model's architecture and hyperparameters.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of type PretrainedConfig.
            ValueError: If the configuration contains invalid values or is incompatible with the model.
        """
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Dense(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, epsilon=1e-12)
        self.vocab_projector = nn.Dense(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Cell:
        """
        Retrieves the output embeddings of the DistilBertForMaskedLM model.

        Args:
            self (DistilBertForMaskedLM): The instance of the DistilBertForMaskedLM class.

        Returns:
            nn.Cell: The output embeddings of the DistilBertForMaskedLM model.
                The embeddings are projected using the vocab_projector.

        Raises:
            None.

        """
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Cell):
        """
        This method sets the output embeddings for the DistilBertForMaskedLM model.

        Args:
            self (DistilBertForMaskedLM): The instance of the DistilBertForMaskedLM class.
            new_embeddings (nn.Cell): The new embeddings to set as the output embeddings for the model.
                It should be an instance of nn.Cell representing the new embeddings to be used.

        Returns:
            None.

        Raises:
            None.
        """
        self.vocab_projector = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[mindspore.Tensor, ...]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):

    """
    DistilBertForSequenceClassification is a class for sequence classification tasks based on the DistilBert architecture.
    This class inherits from the DistilBertPreTrainedModel class and provides a sequence classification layer on top of
    the DistilBert encoder.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        config (PretrainedConfig): The configuration object for the model.
        distilbert (DistilBertModel): The DistilBert encoder model.
        pre_classifier (nn.Dense): A fully connected layer for the classification task.
        classifier (nn.Dense): A fully connected layer for the classification task.
        dropout (nn.Dropout): A dropout layer to prevent overfitting.

    Methods:
        get_position_embeddings: Returns the position embeddings.
        resize_position_embeddings: Resizes the position embeddings of the model.
        construct:
            Constructs the model for sequence classification.

            Args:

            - input_ids (Optional[mindspore.Tensor]): The input sequence tensor of shape `(batch_size, sequence_length)`.
            - attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape `(batch_size, sequence_length)`.
            - head_mask (Optional[mindspore.Tensor]): The head mask tensor of shape `(num_heads,)`.
            - inputs_embeds (Optional[mindspore.Tensor]): The input embeddings tensor of shape `(batch_size, sequence_length, hidden_size)`.
            - labels (Optional[mindspore.Tensor]): The labels tensor of shape `(batch_size,)`.
            - output_attentions (Optional[bool]): Whether to return the attentions tensor or not.
            - output_hidden_states (Optional[bool]): Whether to return the hidden states tensor or not.
            - return_dict (Optional[bool]): Whether to return a dictionary of outputs or a tuple.

            Returns:

            - If `return_dict` is `False`, returns a tuple of `(loss, logits, hidden_states, attentions)`.
            - If `return_dict` is `True`, returns a dictionary of outputs with keys `loss`, `logits`, `hidden_states`, and `attentions`.
    """
    def __init__(self, config: PretrainedConfig):
        """Initialize a DistilBertForSequenceClassification model.

        Args:
            self: The object instance itself.
            config (PretrainedConfig): The configuration object containing various parameters for the model.
                It specifies the model architecture, hyperparameters, and other settings.
                Must be an instance of PretrainedConfig.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not an instance of PretrainedConfig.
            ValueError: If any of the required attributes in the config object are missing or invalid.
            RuntimeError: If there are issues during model initialization or attribute assignment.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Dense(config.dim, config.dim)
        self.classifier = nn.Dense(config.dim, config.num_labels)
        self.dropout = nn.Dropout(p=config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[mindspore.Tensor, ...]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

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
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):

    """
    DistilBertForQuestionAnswering is a class that represents a fine-tuned DistilBERT model for question answering tasks.
    It is designed to provide predictions for the start and end positions of the answer span given a question and a context.

    Please note that this class assumes the existence of a DistilBERT model and a dense layer for question answering
    outputs (qa_outputs), which are initialized in the __init__ method.
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes a new instance of the DistilBertForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config (PretrainedConfig): The configuration object for the pretrained model.

        Returns:
            None.

        Raises:
            ValueError: If the number of labels in the configuration is not equal to 2, a ValueError is raised.
        """
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.qa_outputs = nn.Dense(config.dim, config.num_labels)
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

        self.dropout = nn.Dropout(p=config.qa_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[QuestionAnsweringModelOutput, Tuple[mindspore.Tensor, ...]]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)

        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class DistilBertForTokenClassification(DistilBertPreTrainedModel):

    """
    Represents a DistilBERT model for token classification.

    This class inherits from DistilBertPreTrainedModel and implements methods for initializing the model,
    retrieving position embeddings, resizing position embeddings, and constructing the model for token
    classification tasks.

    Attributes:
        config (PretrainedConfig): The configuration for the pretrained model.

    Methods:
        __init__: Initializes the DistilBertForTokenClassification model.
        get_position_embeddings: Returns the position embeddings.
        resize_position_embeddings: Resizes position embeddings of the model.
        construct: Constructs the model for token classification tasks.

    Raises:
        NotImplementedError: If the method is not implemented.

    Note:
        This class is intended to be subclassed when implementing a custom DistilBERT model for token classification tasks.
    """
    def __init__(self, config: PretrainedConfig):
        """
        Initializes a new instance of the `DistilBertForTokenClassification` class.

        Args:
            self: The object itself.
            config (PretrainedConfig): The configuration for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[mindspore.Tensor, ...]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DistilBertForMultipleChoice(DistilBertPreTrainedModel):

    """
    DistilBertForMultipleChoice is a class that represents a DistilBERT model for multiple choice tasks.
    It is a subclass of DistilBertPreTrainedModel.

    Args:
        config (PretrainedConfig): The configuration class that defines the model architecture and parameters.

    Methods:
        get_position_embeddings:
            Returns the position embeddings.

        resize_position_embeddings:
            Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

            Args:

            - new_num_position_embeddings (`int`): The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size
                will add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.

        construct:
            This method is used to compute the outputs of the model for multiple choice tasks.

            Args:

            - input_ids (Optional[mindspore.Tensor]): The input ids of shape `(batch_size, num_choices, sequence_length)`
            for multiple choice tasks.
            - attention_mask (Optional[mindspore.Tensor]):
            The attention mask of shape `(batch_size, num_choices, sequence_length)`. It is used to avoid performing
            attention on padding token indices.
            - head_mask (Optional[mindspore.Tensor]): The head mask of shape `(num_heads,)`.
            It is used to mask heads of the attention modules.
            - inputs_embeds (Optional[mindspore.Tensor]):
            The embedded input of shape `(batch_size, num_choices, sequence_length, hidden_size)`. It is an alternative
            to input_ids.
            - labels (Optional[mindspore.Tensor]): The labels for computing the multiple choice classification loss.
            Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of
            the input tensors.
            - output_attentions (Optional[bool]): Whether to return attentions tensors or not.
            - output_hidden_states (Optional[bool]): Whether to return hidden states tensors or not.
            - return_dict (Optional[bool]): Whether to return a `MultipleChoiceModelOutput` instead of a tuple.

            Returns:

            If ``return_dict=True``, a :class:`~transformers.MultipleChoiceModelOutput` containing various elements
            depending on the configuration (e.g., ``loss``, ``logits``, ``hidden_states``, ``attentions``),
            otherwise a tuple of objects as follows:

            - **logits** (:obj:`mindspore.Tensor` of shape `(batch_size, num_choices)`): The logits for each choice.
            - **hidden_states** (:obj:`Tuple[mindspore.Tensor]`, optional, returned when ``output_hidden_states=True``
            is passed or when ``config.output_hidden_states=True``): Tuple of :obj:`mindspore.Tensor` of shape
            `(batch_size, sequence_length, hidden_size)`.
            - **attentions** (:obj:`Tuple[mindspore.Tensor]`, optional, returned when ``output_attentions=True``
            is passed or when ``config.output_attentions=True``): Tuple of :obj:`mindspore.Tensor` of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.

    Example:
        ```python
        >>> # importing the required libraries
        >>> from transformers import AutoTokenizer, DistilBertForMultipleChoice
        >>> import torch
        ...
        >>> # loading the tokenizer and model
        >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        >>> model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")
        ...
        >>> # input parameters
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = mindspore.Tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
        ...
        >>> # encoding the prompts and choices
        >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
        >>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1
        ...
        >>> # calculating the loss and logits
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```
    """
    def __init__(self, config: PretrainedConfig):
        """
        __init__

        Initializes a new instance of the DistilBertForMultipleChoice class.

        Args:
            self: The object instance.
            config (PretrainedConfig):
                An instance of PretrainedConfig class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not an instance of PretrainedConfig.
            ValueError: If the config parameter does not contain valid configuration parameters.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Dense(config.dim, config.dim)
        self.classifier = nn.Dense(config.dim, 1)
        self.dropout = nn.Dropout(p=config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`)
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MultipleChoiceModelOutput, Tuple[mindspore.Tensor, ...]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)

        Returns:
            Union[MultipleChoiceModelOutput, Tuple[mindspore.Tensor, ...]]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, DistilBertForMultipleChoice
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
            >>> model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")
            ...
            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> choice0 = "It is eaten with a fork and a knife."
            >>> choice1 = "It is eaten while held in the hand."
            >>> labels = mindspore.Tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
            ...
            >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
            >>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1
            ...
            >>> # the linear classifier still needs to be trained
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        pooled_output = self.dropout(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

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
    "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "DistilBertForMaskedLM",
    "DistilBertForMultipleChoice",
    "DistilBertForQuestionAnswering",
    "DistilBertForSequenceClassification",
    "DistilBertForTokenClassification",
    "DistilBertModel",
    "DistilBertPreTrainedModel",
]
