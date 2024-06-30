# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""MindSpore OpenAI GPT model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import gelu_new, silu
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...ms_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from .configuration_gpt import GPTConfig


logger = logging.get_logger(__name__)


OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-gpt",
    # See all OpenAI GPT models at https://hf-mirror.com/models?filter=openai-gpt
]


ACT_FNS = {"relu": ops.relu, "silu": silu, "gelu": gelu_new, "swish": silu}


class Attention(nn.Cell):

    """
    This class represents an attention mechanism used in neural networks. 
    It is designed to be used as a part of a larger model and inherits from the nn.Cell class.
    
    Attributes:
        bias (Tensor): A tensor representing the bias used in the attention computation.
        n_head (int): The number of attention heads.
        split_size (int): The size of each split in the attention mechanism.
        scale (bool): A flag indicating whether to scale the attention weights.
        c_attn (Conv1D): A 1D convolutional layer used for computing the attention weights.
        c_proj (Conv1D): A 1D convolutional layer used for projecting the attention weights.
        attn_dropout (Dropout): A dropout layer applied to the attention weights.
        resid_dropout (Dropout): A dropout layer applied to the projected attention weights.
        pruned_heads (set): A set of pruned attention heads.

    Methods:
        __init__: Initializes the Attention object.
        prune_heads: Prunes the specified attention heads.
        _attn: Computes the attention weights.
        merge_heads: Merges the attention heads.
        split_heads: Splits the input into multiple attention heads.
        construct: Constructs the attention mechanism.

    Note:
        - The Attention class assumes that the input tensors follow specific shapes and sizes. 
        It is important to ensure that the input data is compatible with the class implementation.
        - The Attention class should be used as part of a larger model and is not intended to be used as a 
        standalone component.
    """
    def __init__(self, nx, n_positions, config, scale=False):
        """
        Initialize the Attention class.

        Args:
            self (Attention): The instance of the Attention class.
            nx (int): The size of the input state.
            n_positions (int): The number of positions.
            config (object): An object containing configuration settings.
            scale (bool): Flag indicating whether to scale the output. Default is False.

        Returns:
            None.

        Raises:
            ValueError: If the input state size (n_state) is not divisible by the number of attention heads specified 
                in the configuration.
        """
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")
        self.bias = ops.tril(ops.ones((n_positions, n_positions))).view(1, 1, n_positions, n_positions)

        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Method: prune_heads

        Description:
            This method prunes the attention heads based on the input 'heads' list and updates the necessary attributes 
            of the Attention class accordingly.

        Args:
            self (Attention): 
                An instance of the Attention class.
                
                - Type: Attention class object
                - Purpose: Represents the current instance of the Attention class where the pruning operation is to 
                be applied.

            heads (list): 
                The list of attention heads to be pruned.
                
                - Type: List
                - Purpose: Contains the indices of the attention heads to be pruned.
                - Restrictions: Should be a non-empty list of integers representing valid attention head indices.

        Returns:
            None:
                - Type: None
                - Purpose: The method does not return any value explicitly but updates the attributes of the Attention 
                class in place.

        Raises:
            None.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, axis=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, axis=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Method _attn in the class Attention calculates attention weights based on the input query (q), key (k),
        and value (v) tensors.

        Args:
            self (Attention): The instance of the Attention class.
            q (Tensor): The input query tensor.
            k (Tensor): The input key tensor.
            v (Tensor): The input value tensor.
            attention_mask (Tensor, optional):
                A mask tensor to mask certain positions in the attention weights. Default is None.
            head_mask (Tensor, optional): A mask tensor to mask certain heads in multi-head attention. Default is None.
            output_attentions (bool, optional): A flag indicating whether to output the attention weights. Default is False.

        Returns:
            outputs (List[Tensor]): A list containing the output tensor representing the weighted values.
                If output_attentions is True, the list also includes the attention weights tensor.

        Raises:
            ValueError: If the dimensionality of the input tensors is not compatible for matrix multiplication.
            ValueError: If the dimensions of the bias tensor are not compatible with the computed attention weights.
            TypeError: If any of the input tensors are not of type Tensor.
            TypeError: If head_mask is provided but not of type Tensor.
            TypeError: If output_attentions is provided but not of type bool.
        """
        w = ops.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.shape[-1])
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.shape[-2], : w.shape[-1]]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = ops.softmax(w, axis=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [ops.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        """
        Merge the heads of the attention mechanism.

        Args:
            self: An instance of the Attention class.
            x: A tensor representing the input data.
                It should have a shape of (batch_size, num_heads, seq_len, head_dim).

        Returns:
            None

        Raises:
            None
        """
        x = x.permute(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(*new_x_shape)  # in Tensorflow implementation: fct merge_states

    def split_heads(self, x, k=False):
        """
        Splits the input tensor into multiple "head" tensors along the last dimension.

        Args:
            self (Attention): The instance of the Attention class.
            x (torch.Tensor): The input tensor to be split into multiple "head" tensors.
                It should have a shape of (batch_size, seq_len, d_model).
            k (bool, optional): A boolean flag indicating whether to transpose the dimensions of the output tensors.
                Default is False.

        Returns:
            torch.Tensor or None: If `k` is True, the function returns a tensor with shape
                (batch_size, seq_len, n_head, d_model/n_head), where the last two dimensions are transposed.
                If `k` is False, the function returns a tensor with shape (batch_size, seq_len, n_head, d_model/n_head).

        Raises:
            None: This method does not raise any exceptions.
        """
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implementation: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        return x.permute(0, 2, 1, 3)

    def construct(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        """
        This method 'construct' in the class 'Attention' processes the input data 'x' through attention mechanisms.

        Args:
            self (object): The instance of the Attention class.
            x (tensor): The input data tensor to be processed.
            attention_mask (tensor, optional):
                An optional mask tensor for masking out certain elements during attention computation.
            head_mask (tensor, optional): An optional mask tensor for masking out specific attention heads.
            output_attentions (bool): A flag indicating whether to output attention weights.

        Returns:
            None: This method does not return any value explicitly;
                it updates internal states and outputs intermediate results.

        Raises:
            ValueError: If the provided 'x' tensor is not compatible for processing.
            RuntimeError: If an error occurs during the attention mechanism computations.
            TypeError: If incorrect data types are provided for the input parameters.
        """
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


class MLP(nn.Cell):

    """
    MLP is a class that represents a multi-layer perceptron (MLP) model.

    MLP is a neural network model that consists of multiple layers of perceptrons or artificial neurons.
    Each layer is fully connected to the next layer, and the final layer produces the output. The MLP class
    inherits from the nn.Cell class, which is a base class for all neural network modules in the PyTorch framework.

    The MLP class has the following attributes:

    - n_state: an integer representing the number of output channels in the first convolutional layer.
    - config: an object containing various configuration parameters for the MLP model.

    The MLP class has the following methods:

    - __init__(self, n_state, config): Initializes the MLP object. It takes two parameters: n_state, which represents
    the number of output channels in the first convolutional layer, and config, which is an object containing
    configuration parameters for the MLP model.
    Inside the method, it initializes the parent class (nn.Cell), sets the number of input channels (nx) to the value
    specified in the config, creates a 1-dimensional convolutional layer (self.c_fc) with n_state output channels and
    nx input channels, creates another 1-dimensional convolutional layer (self.c_proj) with nx output channels and
    n_state input channels, sets the activation function (self.act) to the value specified in the config, and sets the
    dropout probability (self.dropout) to the value specified in the config.
    - construct(self, x): Constructs the MLP model. It takes one parameter, x, which represents the input tensor.
    Inside the method, it applies the activation function to the output of the first convolutional layer (self.c_fc),
    applies the second convolutional layer (self.c_proj) to the result, and returns the output after applying dropout
    (self.dropout).

    Note:
        The MLP class assumes the existence of the ACT_FNS dictionary, which maps activation function names to their
        corresponding functions.

    Example:
        ```python
        >>> config = MLPConfig(n_embd=128, afn='relu', resid_pdrop=0.2)
        >>> mlp = MLP(n_state=64, config=config)
        >>> input_tensor = torch.randn(10, 128)
        >>> output = mlp.construct(input_tensor)
        ```
    """
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        """
        Initializes an instance of the MLP class.

        Args:
            self: The instance of the MLP class.
            n_state (int): Number of states for the MLP.
            config (object): Configuration object containing parameters for the MLP.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, x):
        """
        Constructs the output of the Multi-Layer Perceptron (MLP) model.

        Args:
            self (MLP): The instance of the MLP class.
            x (tensor): The input tensor to be processed by the MLP.

        Returns:
            The constructed output tensor after passing through the MLP layers.

        Raises:
            None.
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Cell):

    """
    This class represents a block in a neural network model.
    It is a subclass of nn.Cell and is used for building transformer models.

    Attributes:
        attn (Attention): The attention module of the block.
        ln_1 (nn.LayerNorm): The first layer normalization module.
        mlp (MLP): The multi-layer perceptron module.
        ln_2 (nn.LayerNorm): The second layer normalization module.

    Methods:
        __init__(self, n_positions, config, scale=False):
            Initializes a new instance of the Block class.

            Args:

            - n_positions (int): The number of positions in the input sequence.
            - config (object): The configuration object for the block.
            - scale (bool, optional): Whether to scale the attention scores. Defaults to False.

        construct(self, x, attention_mask=None, head_mask=None, output_attentions=False):
            Constructs the block by performing the necessary computations on the input.

            Args:

            - x (Tensor): The input tensor.
            - attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            - head_mask (Tensor, optional): The head mask tensor. Defaults to None.
            - output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.

            Returns:

            - outputs (list): A list containing the output tensor and optional attention weights.
    """
    def __init__(self, n_positions, config, scale=False):
        """
        Initializes a Block object.

        Args:
            self (object): The instance of the Block class.
            n_positions (int): The number of positions.
            config (object): The configuration object.
            scale (bool, optional): A flag to indicate scaling. Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm([nx], epsilon=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm([nx], epsilon=config.layer_norm_epsilon)

    def construct(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Constructs a block in the given class.

        Args:
            self (Block): An instance of the Block class.
            x: The input tensor.
            attention_mask (Optional[Tensor]): An optional attention mask tensor. Default is None.
            head_mask (Optional[Tensor]): An optional head mask tensor. Default is None.
            output_attentions (bool): Whether to output attentions. Default is False.

        Returns:
            list: A list containing the output tensor and other optional attentions.

        Raises:
            None.

        This method constructs a block by performing the following steps:

        1. Calculate attention outputs using the 'attn' method, passing the input tensor, attention mask,
           head mask, and output attentions flag as parameters. Store the result in 'attn_outputs'.
        2. Retrieve the first element from 'attn_outputs' and assign it to 'a'.
        3. Add 'x' and 'a' and apply layer normalization using the 'ln_1' method. Store the result in 'n'.
        4. Apply multi-layer perceptron (MLP) to 'n' using the 'mlp' method. Store the result in 'm'.
        5. Add 'n' and 'm' and apply layer normalization using the 'ln_2' method. Store the result in 'h'.
        6. Create a list 'outputs' containing 'h' as the first element, followed by any additional elements
           from 'attn_outputs'.
        7. Return 'outputs'.
        """
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h] + attn_outputs[1:]
        return outputs


class GPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTConfig
    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_unexpected = [r'attn.bias']

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


@dataclass
class GPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`mindspore.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`mindspore.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    mc_loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mc_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class GPTModel(GPTPreTrainedModel):

    """
    This class represents a GPT (Generative Pre-trained Transformer) model for natural language processing tasks.
    It inherits from the GPTPreTrainedModel class and implements the GPT architecture for generating text based on
    input sequences.
    The model includes methods for initializing embeddings, pruning heads, and constructing the model for inference
    or training.

    Attributes:
        config: The configuration for the GPTModel, including parameters such as vocab_size, n_embd, n_positions,
            embd_pdrop, and n_layer.

    Methods:
        __init__: Initializes the GPTModel with the given configuration.
        get_input_embeddings: Returns the input embeddings used in the model.
        set_input_embeddings: Sets new input embeddings for the model.
        _prune_heads): Prunes specified heads of the model based on the provided dictionary
            of layer numbers and heads to prune.
        construct: Constructs the GPTModel for inference or training based on the input data and configuration.
    """
    def __init__(self, config):
        """
        Initializes a GPTModel instance with the provided configuration.

        Args:
            self (GPTModel): The GPTModel instance to be initialized.
            config (object):
                The configuration object containing parameters for the model.

                - vocab_size (int): The size of the vocabulary.
                - n_embd (int): The embedding dimension.
                - n_positions (int): The maximum number of positions.
                - embd_pdrop (float): The dropout probability for embeddings.
                - n_layer (int): The number of layers in the model.

        Returns:
            None.

        Raises:
            TypeError: If config is not of the expected object type.
            ValueError: If any of the configuration parameters are invalid or out of range.
        """
        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])

        self.position_ids = ops.arange(config.n_positions)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the GPTModel.

        Args:
            self: The instance of the GPTModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the GPTModel.

        Args:
            self (GPTModel): An instance of the GPTModel class.
            new_embeddings (object): The new input embeddings to be set.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method allows for updating the input embeddings of the GPTModel by assigning the provided
            'new_embeddings' to the 'tokens_embed' attribute. The 'tokens_embed' attribute is used by the model during
            tokenization and embedding stages.

        The 'self' parameter refers to the current instance of the GPTModel class, while the 'new_embeddings' parameter
        represents the new input embeddings to be assigned. The 'new_embeddings' can be of any data type and should
        contain the updated embeddings.

        Note that the 'tokens_embed' attribute is expected to be updated directly by this method.
        Any existing input embeddings will be overwritten.

        Example:
            ```python
            >>> model = GPTModel()
            >>> new_embeddings = get_new_embeddings()
            >>> model.set_input_embeddings(new_embeddings)
            ```
        """
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutput]:
        '''
        Constructs the GPTModel.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length] containing the input IDs.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor of shape [batch_size, sequence_length] containing the attention mask.
            token_type_ids (Optional[mindspore.Tensor]):
                The token type IDs tensor of shape [batch_size, sequence_length] containing the token type IDs.
            position_ids (Optional[mindspore.Tensor]):
                The position IDs tensor of shape [batch_size, sequence_length] containing the position IDs.
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor of shape [num_heads] containing the head mask.
            inputs_embeds (Optional[mindspore.Tensor]):
                The inputs embeddings tensor of shape [batch_size, sequence_length, hidden_size] containing the input embeddings.
            output_attentions (Optional[bool]):
                Whether to output attentions. If not provided, it uses the value from the configuration.
            output_hidden_states (Optional[bool]):
                Whether to output hidden states. If not provided, it uses the value from the configuration.
            return_dict (Optional[bool]):
                Whether to return a BaseModelOutput instead of a tuple.
                If not provided, it uses the value from the configuration.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutput]:
                The output of the GPTModel.

                - If return_dict is False, it returns a tuple containing the hidden states, all hidden states,
                and all attentions.
                - If return_dict is True, it returns a BaseModelOutput with last_hidden_state, hidden_states,
                and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified at the same time.
            ValueError: If neither input_ids nor inputs_embeds are specified.
        '''
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
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.get_parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * mindspore.tensor(
                np.finfo(mindspore.dtype_to_nptype(self.dtype)).min, attention_mask.dtype)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTLMHeadModel(GPTPreTrainedModel):

    """
    This class represents a language model head for the Generative Pre-trained Transformer (GPT) model.
    It is used for generating language predictions and is designed to be compatible with the GPT architecture.

    The GPTLMHeadModel class provides methods for initializing the model with a configuration, getting and
    setting output embeddings,
    constructing language model outputs, and preparing inputs for generation.
    It inherits from the GPTPreTrainedModel class.

    Methods:
        __init__: Initializes the model with a given configuration.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets new output embeddings for the model.
        construct: Constructs language model outputs based on input features.
        prepare_inputs_for_generation: Prepares input data for language generation.

    The construct method takes various input arguments for language modeling and returns model outputs, including logits
    and hidden states.
    The prepare_inputs_for_generation method prepares input data specifically for language generation tasks.

    Note:
        The GPTLMHeadModel class is designed for use in natural language processing tasks and is a part of the GPT model
        framework.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the GPTLMHeadModel class.

        Args:
            self (GPTLMHeadModel): The instance of the GPTLMHeadModel class.
            config (Config): The configuration object containing model parameters.

        Returns:
            None.

        Raises:
            ValueError: If the configuration object is missing or invalid.
            TypeError: If the configuration object is not of type Config.
            RuntimeError: If there are issues during model initialization.
        """
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the GPTLMHeadModel.

        Args:
            self (GPTLMHeadModel): The instance of the GPTLMHeadModel class.
                This parameter refers to the current instance of the GPTLMHeadModel class.

        Returns:
            The 'lm_head' attribute of the GPTLMHeadModel instance.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the GPTLMHeadModel.

        Args:
            self (object): The instance of the GPTLMHeadModel class.
            new_embeddings (object):
                The new embeddings to be set as the output embeddings for the model.
                It should be of the same type as the existing embeddings.

        Returns:
            None:.

        Raises:
            None
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids: mindspore.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Prepare inputs for generation.

        Args:
            self (GPTLMHeadModel): The instance of the GPTLMHeadModel class.
            input_ids (mindspore.Tensor): The input tensor containing the token ids for the generation.

        Returns:
            Dict[str, Any]: A dictionary containing the prepared input_ids.

        Raises:
            None.
        """
        return {"input_ids": input_ids}


class GPTDoubleHeadsModel(GPTPreTrainedModel):

    """
    This class represents a GPT (Generative Pre-trained Transformer) model with double heads.
    It is used for language modeling and multiple choice classification tasks.
    The GPTDoubleHeadsModel inherits from the GPTPreTrainedModel class.

    The GPTDoubleHeadsModel class contains methods for initializing the model, getting and setting the output embeddings,
    and constructing the model. It also includes a detailed docstring for the `construct` method,
    which describes the input parameters, return values, and provides examples of how to use the method.

    To use the GPTDoubleHeadsModel, follow these steps:

    1. Instantiate the GPTDoubleHeadsModel class, passing the `config` parameter.
    2. Use the `get_output_embeddings` method to get the output embeddings of the model.
    3. Use the `set_output_embeddings` method to set new embeddings for the model.
    4. Use the `construct` method to perform language modeling and multiple choice classification tasks.
    The method takes various input tensors and returns the model outputs, including logits for language
    modeling and multiple choice classification.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, GPTDoubleHeadsModel
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
        >>> model = GPTDoubleHeadsModel.from_pretrained("openai-gpt")
        >>> tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        >>> model.resize_token_embeddings(len(tokenizer))
        ...
        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        >>> input_ids = tokenizer.encode_batch(choices)
        >>> mc_token_ids = [len(ids) - 1 for ids in input_ids]
        ...
        >>> outputs = model.construct(input_ids, mc_token_ids=mc_token_ids)
        >>> lm_logits = outputs.logits
        >>> mc_logits = outputs.mc_logits
        ```
    For more details on how to use the GPTDoubleHeadsModel class, refer to the documentation and examples provided
    in the code.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the GPTDoubleHeadsModel class.

        Args:
            self: The current object.
            config: An instance of the GPTConfig class that holds the configuration settings for the GPT model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        config.num_labels = 1
        self.transformer = GPTModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the GPTDoubleHeadsModel.

        Args:
            self: An instance of the GPTDoubleHeadsModel class.

        Returns:
            None

        Raises:
            None
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the GPTDoubleHeadsModel.

        Args:
            self (GPTDoubleHeadsModel): The instance of the GPTDoubleHeadsModel class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings. This can be an object of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        mc_token_ids: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        mc_labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], GPTDoubleHeadsModelOutput]:
        r"""
        Args:
            mc_token_ids (`mindspore.Tensor` of shape `(batch_size, num_choices)`, *optional*,
                default to index of the last token of the input):
                Index of the classification token in each input sequence. Selected in the range `[0, input_ids.shape[-1] -
                1]`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-1, 0, ..., config.vocab_size]` All labels set to `-100` are
                ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            mc_labels (`mindspore.Tensor` of shape `(batch_size)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
                where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        Return:
            Union[Tuple[mindspore.Tensor], GPTDoubleHeadsModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, OpenAIGPTDoubleHeadsModel
            ...
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
            >>> model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
            >>> tokenizer.add_special_tokens(
            ...     {"cls_token": "[CLS]"}
            ... )  # Add a [CLS] to the vocabulary (we should train it also!)
            >>> model.resize_token_embeddings(len(tokenizer))
            ...
            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
            >>> mc_token_ids = torch.tensor([input_ids.shape[-1] - 1, input_ids.shape[-1] - 1]).unsqueeze(0)  # Batch size 1
            ...
            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        lm_loss, mc_loss = None, None
        if mc_labels is not None:
            mc_loss = ops.cross_entropy(mc_logits.view(-1, mc_logits.shape[-1]), mc_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            lm_loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPTDoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPTForSequenceClassification(GPTPreTrainedModel):

    """
    This class 'GPTForSequenceClassification' represents a sequence classification model based on the GPT
    (Generative Pre-trained Transformer) architecture. It is designed to classify sequences based on the
    provided input.

    This class inherits from the 'GPTPreTrainedModel' class, which provides the basic functionality for a pre-trained
    GPT model.

    The class contains an initializer method '__init__' which takes a 'config' parameter. It calls the initializer of
    the parent class and initializes the 'num_labels' attribute with the 'num_labels' value from the 'config'.
    It also initializes a 'transformer' attribute with an instance of the 'GPTModel' class from the 'config'.
    Additionally, it creates a 'score' attribute which is a neural network layer with a dense layer of shape
    '(config.n_embd, num_labels)' and no bias. Finally, it calls the 'post_init' method.

    The 'construct' method is responsible for constructing the sequence classification model.
    It takes several optional input tensors as parameters, including 'input_ids', 'attention_mask', 'token_type_ids',
    'position_ids', 'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 'output_hidden_states',
    and 'return_dict'. It returns a Tuple of tensors or a 'SequenceClassifierOutput' object.

    The 'labels' parameter is an optional tensor of shape '(batch_size,)', which provides the labels for computing
    the sequence classification/regression loss. The indices in 'labels' should be in the range of
    '[0, ..., config.num_labels - 1]'. If 'config.num_labels == 1', a regression loss (Mean-Square loss) is computed.
    If 'config.num_labels > 1', a classification loss (Cross-Entropy) is computed.

    The 'return_dict' parameter indicates whether the method should return a 'SequenceClassifierOutput' object.
    If 'return_dict' is not provided, it defaults to the value of 'self.config.use_return_dict'.

    The method first calls the 'transformer' model with the provided input tensors and other optional parameters to
    obtain the transformer outputs, including the 'hidden_states' tensor.
    Then, it passes the 'hidden_states' tensor through the 'score' layer to obtain the 'logits' tensor.

    Next, the method checks the shape of the 'input_ids' tensor to determine the batch size. If 'input_ids' is not None,
    the shape of 'input_ids' is used to calculate the sequence lengths. If 'self.config.pad_token_id' is not None,
    the method checks for padding tokens in 'input_ids' and calculates the sequence lengths accordingly.
    If 'input_ids' is None, the sequence lengths are set to -1.

    The method then selects the relevant logits based on the sequence lengths.
    If 'sequence_lengths' is an integer, the method uses it to index the 'logits' tensor. Otherwise,
    it uses the 'sequence_lengths' tensor to gather the relevant logits.

    The 'loss' variable is set to None initially. If 'labels' is provided, the method determines the 'problem_type'
    based on the 'config' and the shape and dtype of 'labels'. Depending on the 'problem_type', the method calculates
    the loss using operations provided by the 'ops' module.

    Finally, depending on the 'return_dict' parameter, the method either returns a Tuple of tensors or a
    'SequenceClassifierOutput' object containing the 'loss', 'logits', 'hidden_states', and 'attentions'.

    Note:
        This docstring does not include method signatures or any other code for clarity.
    """
    def __init__(self, config):
        """
        Initializes an instance of GPTForSequenceClassification.

        Args:
            self (GPTForSequenceClassification): The instance itself.
            config:
                An object containing configuration settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be compatible with the GPTModel configuration.

        Returns:
            None.

        Raises:
            NotImplementedError: If the method 'post_init' is not implemented in the derived class.
            TypeError: If the configuration settings provided are not compatible with the GPTModel.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.score = nn.Dense(config.n_embd, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        # Ensure the batch size is > 1 if there is no padding.
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                # avoid backward error
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        if isinstance(sequence_lengths, int):
            pooled_logits = logits[ops.arange(batch_size), sequence_lengths]
        else:
            pooled_logits = ops.gather(logits, sequence_lengths, 1, 1)

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
                    loss = ops.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

__all__ = [
    "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "GPTDoubleHeadsModel",
    "GPTForSequenceClassification",
    "GPTLMHeadModel",
    "GPTModel",
    "GPTPreTrainedModel",
]
