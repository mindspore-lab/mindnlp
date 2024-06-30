# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Bark
"""
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal


from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...generation.logits_process import (
    AlternatingCodebooksLogitsProcessor,
    BarkEosPrioritizerLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from ..auto import AutoModel
from .configuration_bark import (
    BarkCoarseConfig,
    BarkConfig,
    BarkFineConfig,
    BarkSemanticConfig,
    BarkSubModelConfig,
)
from .generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "suno/bark-small"
_CONFIG_FOR_DOC = "BarkConfig"

BARK_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "suno/bark-small",
    "suno/bark",
    # See all Bark models at https://hf-mirror.com/models?filter=bark
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    '''
    This function retrieves unpad data from the attention_mask.
    
    Args:
        attention_mask (Tensor): A 2D tensor representing the attention mask for the input data.
            Its shape is (batch_size, sequence_length), where batch_size corresponds to the number of input sequences
            and sequence_length corresponds to the maximum sequence length in the batch.

    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A 1D tensor containing the indices of non-zero elements in the flattened attention mask tensor.
            - cu_seqlens (Tensor): A 1D tensor representing the cumulative sum of sequence lengths in the batch, padded with a zero at the beginning.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None
    '''
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class BarkSelfAttention(nn.Cell):

    """
    Represents a self-attention mechanism for the Bark model.

    This class inherits from nn.Cell and implements a self-attention mechanism for the Bark model.
    It includes methods for splitting and merging heads, performing attention calculations, and constructing the
    self-attention mechanism.

    Attributes:
        config: Configuration settings for the self-attention mechanism.
        is_causal (bool): Flag indicating whether the self-attention mechanism is causal.

    Methods:
        __init__(self, config, is_causal=False): Initializes the BarkSelfAttention class with the specified configuration and causal flag.
        _split_heads(self, tensor, num_heads, attn_head_size): Splits the hidden_size dimension into attn_head_size and num_heads.
        _merge_heads(self, tensor, num_heads, attn_head_size): Merges the attn_head_size dimension and num_attn_heads dimension into hidden_size.
        _attn(self, query, key, value, attention_mask=None, head_mask=None):
            Performs the attention calculation using the provided query, key, value, attention_mask, and head_mask.
        construct(self, hidden_states, attention_mask=None, past_key_values=None, head_mask=None, use_cache=False, output_attentions=False):
            Constructs the self-attention mechanism using the specified parameters and returns the outputs.

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.

    Note:
        Always use triple double quotes around docstrings for consistency.
    """
    # adapted from GPTNeoSelfAttention and Bark code
    # BarkSelfAttention can have two attention type, i.e full attention or causal attention

    def __init__(self, config, is_causal=False):
        """
        Initialize the BarkSelfAttention class.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the self-attention mechanism.
                This parameter is required and must be of a specific format.
            is_causal: A boolean flag indicating whether the self-attention mechanism should be causal.
               Defaults to False if not provided.

        Returns:
            None.

        Raises:
            ValueError: If the `embed_dim` is not divisible by `num_heads`, an exception is raised with a specific message.
        """
        super().__init__()

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.resid_dropout = nn.Dropout(p=config.dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # key, query, value projections for all heads, but in a batch
        self.att_proj = nn.Dense(config.hidden_size, 3 * config.hidden_size, has_bias=config.bias)
        # output projection
        self.out_proj = nn.Dense(config.hidden_size, config.hidden_size, has_bias=config.bias)

        self.is_causal = is_causal
        if is_causal:
            block_size = config.block_size
            bias = ops.tril(ops.ones((block_size, block_size), dtype=mindspore.bool_)).view(1, 1, block_size, block_size)
            self.bias = Parameter(bias, requires_grad=False)

    # Copied from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._split_heads
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # re-assemble all head outputs side by side
        # (batch, num_heads, seq_len, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.swapaxes(1, 2)
        tensor = tensor.view(tensor.shape[:-2] + (num_heads * attn_head_size,))

        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Performs self-attention mechanism on the given query, key, and value tensors.

        Args:
            self (BarkSelfAttention): The instance of the BarkSelfAttention class.
            query (Tensor): The query tensor of shape (batch_size, num_heads, sequence_length, head_dim).
            key (Tensor): The key tensor of shape (batch_size, num_heads, sequence_length, head_dim).
            value (Tensor): The value tensor of shape (batch_size, num_heads, sequence_length, head_dim).
            attention_mask (Tensor, optional): The attention mask tensor of shape (batch_size, 1, sequence_length, sequence_length). Defaults to None.
            head_mask (Tensor, optional): The head mask tensor of shape (num_heads, 1, 1, sequence_length). Defaults to None.

        Returns:
            attn_output (Tensor): The attention output tensor of shape (batch_size, num_heads, sequence_length, head_dim).
            attn_weights (Tensor): The attention weights tensor of shape (batch_size, num_heads, sequence_length, sequence_length).

        Raises:
            None.
        """
        # unlike GPTNeo's SelfAttention, divide by the square root of the dimension of the query and the key
        attn_weights = ops.matmul(query, key.swapaxes(-1, -2)) * (1.0 / math.sqrt(self.head_dim))

        if self.is_causal:
            query_length, key_length = query.shape[-2], key.shape[-2]

            # fill the upper left part of the attention weights with inf
            attn_weights = attn_weights.masked_fill(
                self.bias[:, :, key_length - query_length : key_length, :key_length] == 0,
                finfo(attn_weights.dtype, 'min'),
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, attn_head_size)
        # -> (batch, num_heads, seq_len, attn_head_size)
        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        This method constructs the self-attention mechanism in the BarkSelfAttention class.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The input hidden states to be used in the attention mechanism.
            attention_mask (Tensor, optional): Mask to prevent attention to certain positions. Default is None.
            past_key_values (Tuple, optional): Tuple containing past key and value tensors for incremental decoding. Default is None.
            head_mask (Tensor, optional): Mask to prevent attention to certain heads. Default is None.
            use_cache (bool, optional): Flag indicating whether to use cache for incremental decoding. Default is False.
            output_attentions (bool, optional): Flag indicating whether to output attention weights. Default is False.

        Returns:
            outputs (Tuple):
                A tuple containing the attention output tensor and present key-value tuple.
                If output_attentions is True, the tuple also includes attention weights.
                Returns None if no output is generated.

        Raises:
            ValueError: If the dimensions of the input tensors are not compatible for the attention mechanism.
            TypeError: If any of the input arguments are of incorrect type.
            IndexError: If the past_key_values tuple does not contain expected elements.
        """
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, axis=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if past_key_values is not None:
            past_key = past_key_values[0]
            past_value = past_key_values[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

BARK_ATTENTION_CLASSES = {
    "eager": BarkSelfAttention,
}


class BarkLayerNorm(nn.Cell):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""
    def __init__(self, hidden_size, bias=True):
        """
        The __init__ method initializes an instance of the BarkLayerNorm class.

        Args:
            self: The instance of the class.
            hidden_size (int): The size of the hidden layer for the neural network.
            bias (bool): A flag to determine whether to include bias in the layer normalization. Defaults to True.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.weight = Parameter(ops.ones((hidden_size,)))
        self.bias = Parameter(ops.zeros((hidden_size,))) if bias else ops.zeros((hidden_size,))
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        epsilon=1e-5)
    def construct(self, inputs):
        """
        Constructs a normalized layer in the BarkLayerNorm class.

        Args:
            self (BarkLayerNorm): An instance of the BarkLayerNorm class.
            inputs (Any): The input data to be normalized.

        Returns:
            None: The method modifies the inputs in-place.

        Raises:
            TypeError: If the inputs are not compatible with the normalization process.

        This method normalizes the inputs using the layer norm technique and modifies them in-place.
        The normalization process involves computing the mean and standard deviation of the inputs and then scaling
        and shifting them using learned parameters. The normalized inputs are returned as output.

        Please note that this method assumes the inputs are in the correct shape and format for normalization. Any incompatible inputs will raise a TypeError.
        """
        y, _, _ = self.layer_norm(inputs, self.weight, self.bias)
        return y

class BarkMLP(nn.Cell):

    """
    BarkMLP represents a multi-layer perceptron (MLP) neural network architecture implemented in MindSpore,
    utilizing dense layers, dropout, and GELU activation function.

    Attributes:
        in_proj (nn.Dense): The input projection layer of the MLP, mapping input features to a higher-dimensional space.
        out_proj (nn.Dense): The output projection layer of the MLP, mapping the higher-dimensional space back to the original feature space.
        dropout (nn.Dropout): A dropout layer to regularize the network by randomly setting a fraction of input units to zero.
        gelu (nn.GELU): The Gaussian Error Linear Unit (GELU) activation function applied to introduce non-linearity.

    Methods:
        construct(hidden_states): Constructs the forward pass of the MLP by sequentially passing the input through
        the input projection, GELU activation, output projection, and dropout layers.

    Note:
        The 'BarkMLP' class inherits from 'nn.Cell' for compatibility with MindSpore neural network modules.
    """
    def __init__(self, config):
        """
        Initializes a BarkMLP instance.

        Args:
            self: The instance of the BarkMLP class.
            config:
                An object containing configuration parameters for the MLP model.

                - Type: Custom class
                - Purpose: Specifies the configuration settings for the MLP model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            TypeError: If the configuration parameters are not provided in the expected format.
            ValueError: If there are issues with the configuration values, such as invalid sizes or types.
            RuntimeError: If there are errors during the initialization process of the model components.
        """
        super().__init__()
        self.in_proj = nn.Dense(config.hidden_size, 4 * config.hidden_size, has_bias=config.bias)
        self.out_proj = nn.Dense(4 * config.hidden_size, config.hidden_size, has_bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)
        self.gelu = nn.GELU(approximate=False)

    def construct(self, hidden_states):
        """
        Constructs the hidden states by applying a series of transformations.

        Args:
            self (BarkMLP): The instance of the BarkMLP class.
            hidden_states (tensor): The input hidden states to be processed.

        Returns:
            tensor: The processed hidden states after applying the series of transformations.

        Raises:
            ValueError: If the input hidden states are not in the expected format.
            RuntimeError: If an error occurs during the transformation process.
        """
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BarkBlock(nn.Cell):

    """
    BarkBlock represents a building block for a neural network model,
    specifically designed for handling attention mechanisms and MLP layers.
    This class inherits from nn.Cell and consists of methods for initializing the block
    and constructing the block's forward pass.

    Attributes:
        layernorm_1: An instance of either BarkLayerNorm or nn.LayerNorm based on the 'is_causal' flag.
        layernorm_2: An instance of either BarkLayerNorm or nn.LayerNorm based on the 'is_causal' flag.
        attn: An instance of an attention mechanism chosen from the BARK_ATTENTION_CLASSES dictionary.
        mlp: An instance of the BarkMLP class.

    Methods:
        __init__(self, config, is_causal=False): Initializes the BarkBlock instance with the given configuration and causal flag.
        construct(self, hidden_states, past_key_values=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
            Constructs the forward pass of the block using the provided inputs and optional arguments.

    Example:
        ```python
        >>> config = Configuration(hidden_size=512, bias=True)
        >>> block = BarkBlock(config, is_causal=True)
        >>> hidden_states = torch.randn(1, 10, 512)
        >>> outputs = block.construct(hidden_states, attention_mask=torch.ones(1, 10))
        ```
    """
    def __init__(self, config, is_causal=False):
        """
        Initializes a new instance of BarkBlock.

        Args:
            self: The instance of the class.
            config: An object representing the configuration settings.
            is_causal: A boolean indicating whether the attention is causal or not.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()

        if is_causal:
            # if causal, uses handmade LayerNorm, so that the layerNorm bias is optional
            # this handmade layerNorm is used to stick with Bark choice of leaving optional bias in
            # AutoRegressive models (corresponding to the "Text" and the "Coarse" modules)
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)

        self.attn = BARK_ATTENTION_CLASSES["eager"](config, is_causal=is_causal)

        self.mlp = BarkMLP(config)

    def construct(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        This method constructs a BarkBlock by processing the given hidden states through attention mechanisms.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The input hidden states to be processed.
            past_key_values (Tuple, optional): Tuple containing past key values for optimization.
            attention_mask (Tensor, optional): Masking tensor to prevent attention to certain positions.
            head_mask (Tensor, optional): Masking tensor to control which heads are active in the attention computation.
            use_cache (bool): Flag indicating whether to use caching for intermediate computations.
            output_attentions (bool): Flag indicating whether to output attention weights.

        Returns:
            Tuple: Returns a tuple containing the updated intermediary hidden states and any additional outputs.

        Raises:
            TypeError: If the input types are incorrect.
            ValueError: If the input values are invalid.
        """
        intermediary_hidden_states = self.layernorm_1(hidden_states)

        attn_outputs = self.attn(
            intermediary_hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attn_output = attn_outputs[0]  # output_attn: output, present_key_values, (attn_weights)
        outputs = attn_outputs[1:]

        intermediary_hidden_states = hidden_states + attn_output
        intermediary_hidden_states = intermediary_hidden_states + self.mlp(
            self.layernorm_2(intermediary_hidden_states)
        )

        if use_cache:
            outputs = (intermediary_hidden_states,) + outputs
        else:
            outputs = (intermediary_hidden_states,) + outputs[1:]

        return outputs  # hidden_states, ((present), attentions)


class BarkPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BarkConfig
    supports_gradient_checkpointing = False

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

# GPT2-like autoregressive model
class BarkCausalModel(BarkPreTrainedModel):

    """
    The `BarkCausalModel` class is a subclass of `BarkPreTrainedModel` and represents a model
    for causal language modeling using the Bark framework.

    Attributes:
        `config`: An instance of the `BarkConfig` class containing the model configuration.
        `input_embeds_layer`: An embedding layer for the input vocabulary.
        `position_embeds_layer`: An embedding layer for the position indices.
        `drop`: A dropout layer.
        `layers`: A list of `BarkBlock` layers for the model.
        `layernorm_final`: A layer normalization module for the final hidden states.
        `lm_head`: A dense layer for generating the output vocabulary logits.
        `gradient_checkpointing`: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        `__init__(self, config)`: Initializes the `BarkCausalModel` instance.
        `get_input_embeddings(self)`: Returns the input embedding layer.
        `set_input_embeddings(self, new_embeddings)`: Sets the input embedding layer.
        `prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs)`: Prepares the inputs for generation.
        `construct(self, input_ids, past_key_values=None, attention_mask=None, position_ids=None, head_mask=None, labels=None, input_embeds=None, use_cache=None, output_attentions=None,
            output_hidden_states=None, return_dict=None)`: Constructs the model output based on the provided inputs.
        `_reorder_cache(past_key_values, beam_idx)`: Reorders the cache for beam search or beam sampling.

    Note:
        This docstring provides an overview of the class and its methods.
        For detailed information on each method, please refer to the corresponding method's docstring.
    """
    config_class = BarkSubModelConfig

    def __init__(self, config):
        """
        Initializes an instance of the BarkCausalModel class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration parameters for the model.

                - input_vocab_size (int): The size of the input vocabulary.
                - hidden_size (int): The size of the hidden state.
                - block_size (int): The size of the block.
                - dropout (float): The dropout probability.
                - num_layers (int): The number of layers.
                - bias (bool): Whether to apply bias in BarkLayerNorm.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        # initialize as an autoregressive GPT-like model
        self.input_embeds_layer = nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        self.drop = nn.Dropout(p=config.dropout)

        self.layers = nn.CellList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])

        self.layernorm_final = BarkLayerNorm(config.hidden_size, bias=config.bias)

        self.lm_head = nn.Dense(config.hidden_size, config.output_vocab_size, has_bias=False)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the BarkCausalModel.

        Args:
            self (BarkCausalModel): The instance of the BarkCausalModel class.

        Returns:
            None: This method returns the input embeddings layer of the BarkCausalModel.

        Raises:
            This method does not raise any exceptions.
        """
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        """
        Set input embeddings for the BarkCausalModel.

        Args:
            self (BarkCausalModel): The instance of BarkCausalModel.
            new_embeddings (any): The new input embeddings to be set for the model. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.input_embeds_layer = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        This method prepares inputs for generation in the BarkCausalModel class.

        Args:
            self (object): The instance of the class.
            input_ids (tensor): The input tensor containing the tokenized input sequence.
            past_key_values (tuple, optional): The past key values for fast decoding. Defaults to None.

        Returns:
            dict or None: A dictionary containing the prepared input values for generation,
            including the input_ids, input_embeds, past_key_values, use_cache, position_ids, and attention_mask.
            Returns None if input_embeds is not provided and use_cache is False.

        Raises:
            ValueError: If the input_ids shape is incompatible with past_key_values.
            ValueError: If the input_embeds shape is incompatible with use_cache.
            TypeError: If the attention_mask and position_ids shapes are not compatible.
            RuntimeError: If there are issues with calculating position_ids based on attention_mask.
        """
        input_embeds = kwargs.get("input_embeds", None)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if past_key_values is not None:
            # Omit tokens covered by past_key_values
            seq_len = input_ids.shape[1]
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            # input_embeds have already been used and is not required anymore
            input_embeds = None
        else:
            if input_embeds is not None and kwargs.get("use_cache"):
                seq_len = input_embeds.shape[1]
            else:
                seq_len = input_ids.shape[1]

        # ensure that attention_mask and position_ids shapes are aligned with the weird Bark hack of reducing
        # sequence length on the first forward pass
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        if position_ids is not None:
            position_ids = position_ids[:, :seq_len]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        if input_embeds is not None and kwargs.get("use_cache"):
            return {
                "input_ids": None,
                "input_embeds": input_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        input_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithPast]:
        '''
        Constructs the BarkCausalModel.

        Args:
            self: The object itself.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length). Defaults to None.
            past_key_values (Optional[Tuple[mindspore.Tensor]]):
                The past key values tensor of shape (batch_size, num_heads, sequence_length, embed_size_per_head).
                Defaults to None.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (Optional[mindspore.Tensor]):
                The position ids tensor of shape (batch_size, sequence_length). Defaults to None.
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor of shape (num_heads, sequence_length, sequence_length). Defaults to None.
            labels (Optional[mindspore.Tensor]):
                The labels tensor of shape (batch_size, sequence_length). Defaults to None.
            input_embeds (Optional[mindspore.Tensor]):
                The input embeddings tensor of shape (batch_size, sequence_length, embed_size). Defaults to None.
            use_cache (Optional[bool]): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool]): Whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], CausalLMOutputWithPast]:
                The output of the model. It can be a tuple containing the following elements:

                - loss (mindspore.Tensor): The loss tensor.
                - logits (mindspore.Tensor): The logits tensor.
                - past_key_values (Tuple[mindspore.Tensor]): The past key values tensor.
                - hidden_states (Tuple[mindspore.Tensor]): The hidden states tensor.
                - attentions (Tuple[mindspore.Tensor]): The attentions tensor.
            or an instance of the CausalLMOutputWithPast class.

        Raises:
            ValueError: If both input_ids and input_embeds are specified.
            ValueError: If batch_size is not defined or <= 0.
            NotImplementedError: If training is not implemented yet for Bark - ensure you do not pass labels to the model.
            '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Verify if input_embeds already exists
        # then compute embeddings.
        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        if input_embeds is not None and past_key_values is None:
            # we want to return the input_embeds in priority so that it is in line with a weird hack
            # of Bark which concatenate two bits of the input_embeds on the first forward pass of the semantic model
            pass
        elif input_ids is not None:
            input_embeds = self.input_embeds_layer(input_ids)  # token embeddings of shape (b, t, n_embd)
        elif input_embeds is not None:
            pass
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")

        input_shape = input_embeds.shape[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[-1]

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = ops.arange(past_length, seq_length + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)  # shape (1, seq_length)

        position_embeds = self.position_embeds_layer(position_ids)  # position embeddings of shape (1, t, n_embd)

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # [bsz, to_seq_length] -> [bsz, 1, 1, to_seq_length]
            # from_seq_length is 1 to easily broadcast
            attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape num_layers x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.shape[-1],)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        present_key_values = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, past_layer_key_values) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                past_key_values=past_layer_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if use_cache:
                present_key_values = present_key_values + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.layernorm_final(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError(
                "Training is not implemented yet for Bark - ensure you do not pass `labels` to the model."
            )

        if not return_dict:
            return tuple(
                v for v in [None, logits, present_key_values, all_hidden_states, all_self_attentions] if v is not None
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[mindspore.Tensor]], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # Necessary for beam_search
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )


class BarkSemanticModel(BarkCausalModel):

    """
        Represents a semantic model for generating text semantic tokens from an input prompt and an optional `Bark` speaker prompt.

        This class inherits from BarkCausalModel and provides a method to generate output semantic tokens
        based on the input prompt and generation configuration.

        Attributes:
            input_embeds_layer (Layer): The layer used for input embeddings.
            config (Config): Configuration settings for the semantic model.
        """
    base_model_prefix = "semantic"
    config_class = BarkSemanticConfig

    def generate(
        self,
        input_ids: mindspore.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        history_prompt: Optional[Dict[str, mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> mindspore.Tensor:
        """
        Generates text semantic tokens from an input prompt and an additional optional `Bark` speaker prompt.

        Args:
            input_ids (`Optional[mindspore.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids, i.e tokenized input sentences. Will be truncated up to
                semantic_generation_config.max_input_semantic_length tokens. Note that the output audios will be as
                long as the longest generation among the batch.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            history_prompt (`Optional[Dict[str,mindspore.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
            attention_mask (`Optional[mindspore.Tensor]`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

        Returns:
            mindspore.Tensor: Output semantic tokens.
        """
        if semantic_generation_config is None:
            raise ValueError("`semantic_generation_config` has to be provided")

        batch_size = input_ids.shape[0]

        max_input_semantic_length = semantic_generation_config.max_input_semantic_length

        input_ids = input_ids + semantic_generation_config.text_encoding_offset

        if attention_mask is not None:
            input_ids = input_ids.masked_fill((1 - attention_mask).bool(), semantic_generation_config.text_pad_token)

        if history_prompt is not None:
            semantic_history = history_prompt["semantic_prompt"][-max_input_semantic_length:]
            semantic_history = ops.pad(
                semantic_history,
                (0, max_input_semantic_length - len(semantic_history)),
                value=semantic_generation_config.semantic_pad_token,
                mode="constant",
            )
        else:
            semantic_history = mindspore.Tensor(
                [semantic_generation_config.semantic_pad_token] * max_input_semantic_length, dtype=mindspore.int32
            )

        semantic_history = ops.repeat_interleave(semantic_history[None], batch_size, axis=0)

        infer_array = mindspore.Tensor(
            [[semantic_generation_config.semantic_infer_token]] * batch_size, dtype=mindspore.int32
        )

        input_embeds = ops.cat(
            [
                self.input_embeds_layer(input_ids[:, :max_input_semantic_length])
                + self.input_embeds_layer(semantic_history[:, : max_input_semantic_length + 1]),
                self.input_embeds_layer(infer_array),
            ],
            axis=1,
        )

        tokens_to_suppress = list(
            range(semantic_generation_config.semantic_vocab_size, semantic_generation_config.semantic_pad_token)
        )
        tokens_to_suppress.extend(
            list(range(semantic_generation_config.semantic_pad_token + 1, self.config.output_vocab_size))
        )

        suppress_tokens_logits_processor = SuppressTokensLogitsProcessor(tokens_to_suppress)

        min_eos_p = kwargs.get("min_eos_p", semantic_generation_config.min_eos_p)
        early_stopping_logits_processor = BarkEosPrioritizerLogitsProcessor(
            eos_token_id=semantic_generation_config.eos_token_id, min_eos_p=min_eos_p
        )

        # pass input_ids in order to stay consistent with the transformers generate method even though it is not used
        # (except to get the input seq_len - that's why we keep the first 257 tokens)
        semantic_output = super().generate(
            ops.ones((batch_size, max_input_semantic_length + 1), dtype=mindspore.int64),
            input_embeds=input_embeds,
            logits_processor=[suppress_tokens_logits_processor, early_stopping_logits_processor],
            generation_config=semantic_generation_config,
            **kwargs,
        )  # size: 10048

        # take the generated semantic tokens
        semantic_output = semantic_output[:, max_input_semantic_length + 1 :]

        return semantic_output


class BarkCoarseModel(BarkCausalModel):

    """
    Represents a model for generating coarse acoustics tokens from input text semantic tokens and an optional `Bark` speaker prompt.

    This class inherits from BarkCausalModel and includes methods for preprocessing histories and generating coarse acoustics tokens based on provided configurations and inputs.

    Methods:
        preprocess_histories(max_coarse_history, semantic_to_coarse_ratio, batch_size, semantic_generation_config, codebook_size, history_prompt=None):
            Preprocesses optional `Bark` speaker prompts before generating coarse acoustics tokens. Returns processed semantic and coarse speaker prompts.

        generate(semantic_output, semantic_generation_config, coarse_generation_config, codebook_size=1024, history_prompt=None, return_output_lengths=None, **kwargs):
            Generates coarse acoustics tokens based on input text semantic tokens, generation configurations, and optional speaker prompts. Returns the output coarse acoustics tokens.

    Args:
        semantic_output (mindspore.Tensor): Input text semantic ids.
        semantic_generation_config (BarkSemanticGenerationConfig): Generation config for semantic tokens.
        coarse_generation_config (BarkCoarseGenerationConfig): Generation config for coarse tokens.
        codebook_size (int, optional): Size of the output vocabulary per codebook channel.
        history_prompt (Optional[Dict[str, mindspore.Tensor]], optional): Optional `Bark` speaker prompt.
        return_output_lengths (bool, optional): Whether to return the output lengths.

    Returns:
        Conditional return:
            By default:

            - mindspore.Tensor: Output coarse acoustics tokens.

            If return_output_lengths=True:

            - Tuple(mindspore.Tensor, mindspore.Tensor):
            Output coarse acoustics tokens and the length of each sample in the batch.
    """
    base_model_prefix = "coarse_acoustics"
    config_class = BarkCoarseConfig

    def preprocess_histories(
        self,
        max_coarse_history: int,
        semantic_to_coarse_ratio: int,
        batch_size: int,
        semantic_generation_config: int,
        codebook_size: int,
        history_prompt: Optional[Dict[str, mindspore.Tensor]] = None,
    ):
        """
        Preprocess the optional `Bark` speaker prompts before `self.generate`.

        Args:
            max_coarse_history (`int`):
                Maximum size of coarse tokens used.
            semantic_to_coarse_ratio (`int`):
                Ratio of semantic to coarse frequency
            batch_size (`int`):
                Batch size, i.e the number of samples.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            codebook_size (`int`):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,mindspore.Tensor]]`):
                Optional `Bark` speaker prompt.

        Returns:
            `tuple(mindspore.Tensor)`:

                - **x_semantic_history** (`mindspore.Tensor` -- Processed semantic speaker prompt.
                - **x_coarse_history** (`mindspore.Tensor`) -- Processed coarse speaker prompt.
        """
        if history_prompt is not None:
            x_semantic_history = ops.repeat_interleave(history_prompt["semantic_prompt"][None], batch_size, axis=0)
            # clone to avoid modifying history_prompt.coarse_prompt
            x_coarse_history = history_prompt["coarse_prompt"].copy()

            # offset x_coarse_history
            if codebook_size is not None:
                for n in range(1, x_coarse_history.shape[0]):
                    # offset
                    x_coarse_history[n, :] += codebook_size * n

            # flatten x_coarse_history
            x_coarse_history = ops.swapaxes(x_coarse_history, 0, 1).view(-1)

            x_coarse_history = x_coarse_history + semantic_generation_config.semantic_vocab_size

            x_coarse_history = ops.repeat_interleave(x_coarse_history[None], batch_size, axis=0)
            # e.g: after SEMANTIC_VOCAB_SIZE (10000), 1024 tokens dedicated to first codebook, 1024 next tokens
            # dedicated to second codebook.

            max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
            # trim histories correctly
            n_semantic_hist_provided = min(
                [
                    max_semantic_history,
                    x_semantic_history.shape[1] - x_semantic_history.shape[1] % 2,
                    int(np.floor(x_coarse_history.shape[1] / semantic_to_coarse_ratio)),
                ]
            )

            n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))

            x_semantic_history = x_semantic_history[:, -n_semantic_hist_provided:].long()
            x_coarse_history = x_coarse_history[:, -n_coarse_hist_provided:].long()
            # bit of a hack for time alignment (sounds better) - from Bark original implementation
            x_coarse_history = x_coarse_history[:, :-2]

        else:
            # shape: (batch_size, 0)
            x_semantic_history = mindspore.tensor(mindspore._c_expression.Tensor([[]] * batch_size, mindspore.int64))
            x_coarse_history = mindspore.tensor(mindspore._c_expression.Tensor([[]] * batch_size, mindspore.int64))
        return x_semantic_history, x_coarse_history

    def generate(
        self,
        semantic_output: mindspore.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        coarse_generation_config: BarkCoarseGenerationConfig = None,
        codebook_size: int = 1024,
        history_prompt: Optional[Dict[str, mindspore.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,
    ) -> Union[mindspore.Tensor, Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Generates coarse acoustics tokens from input text semantic tokens and an additional optional `Bark` speaker
        prompt.

        Args:
            semantic_output (`mindspore.Tensor` of shape (batch_size, seq_len), *optional*):
                Input text semantic ids, i.e the output of `BarkSemanticModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,mindspore.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
            return_output_lengths (`bool`, *optional*):
                Whether or not to return the output lengths. Useful when batching.

        Returns:
            Conditional return:
                By default:

                - mindspore.Tensor: Output coarse acoustics tokens.
                If `return_output_lengths=True`:

                - `Tuple(mindspore.Tensor, mindspore.Tensor): The output coarse acoustics tokens, and the
                length of each sample of the batch.
        """
        if semantic_generation_config is None:
            raise ValueError("`semantic_generation_config` has to be provided")

        if coarse_generation_config is None:
            raise ValueError("`coarse_generation_config` has to be provided")

        max_coarse_input_length = coarse_generation_config.max_coarse_input_length
        max_coarse_history = coarse_generation_config.max_coarse_history
        sliding_window_len = coarse_generation_config.sliding_window_len

        # replace semantic_pad_token (eos_tok and pad_tok here) with coarse_semantic_pad_token i.e the pad_token
        # used in the next model
        semantic_output = semantic_output.masked_fill(
            semantic_output == semantic_generation_config.semantic_pad_token,
            coarse_generation_config.coarse_semantic_pad_token,
        )

        semantic_to_coarse_ratio = (
            coarse_generation_config.coarse_rate_hz
            / semantic_generation_config.semantic_rate_hz
            * coarse_generation_config.n_coarse_codebooks
        )
        max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))

        output_lengths = (semantic_output != coarse_generation_config.coarse_semantic_pad_token).sum(1)
        output_lengths = ops.floor(
            output_lengths * semantic_to_coarse_ratio / coarse_generation_config.n_coarse_codebooks
        )
        output_lengths = ops.round(output_lengths * coarse_generation_config.n_coarse_codebooks).int()

        max_generated_len = ops.max(output_lengths)[0].item()

        batch_size = semantic_output.shape[0]

        x_semantic_history, x_coarse = self.preprocess_histories(
            history_prompt=history_prompt,
            max_coarse_history=max_coarse_history,
            semantic_to_coarse_ratio=semantic_to_coarse_ratio,
            batch_size=batch_size,
            semantic_generation_config=semantic_generation_config,
            codebook_size=codebook_size,
        )
        base_semantic_idx = x_semantic_history.shape[1]

        semantic_output = ops.hstack([x_semantic_history, semantic_output])

        n_window_steps = int(np.ceil(max_generated_len / sliding_window_len))

        total_generated_len = 0

        len_coarse_history = x_coarse.shape[1]

        for _ in range(n_window_steps):
            semantic_idx = base_semantic_idx + int(round(total_generated_len / semantic_to_coarse_ratio))

            # pad from right side
            input_coarse = semantic_output[:, int(np.max([0, semantic_idx - max_semantic_history])) :]
            input_coarse = input_coarse[:, :max_coarse_input_length]
            input_coarse = ops.pad(
                input_coarse,
                (0, max_coarse_input_length - input_coarse.shape[-1]),
                "constant",
                coarse_generation_config.coarse_semantic_pad_token,
            )

            if 0 in x_coarse.shape:
                input_coarse = ops.hstack(
                    [
                        input_coarse,
                        mindspore.Tensor([[coarse_generation_config.coarse_infer_token]] * batch_size),
                    ]
                )
            else:
                input_coarse = ops.hstack(
                    [
                        input_coarse,
                        mindspore.Tensor([[coarse_generation_config.coarse_infer_token]] * batch_size),
                        x_coarse[:, -max_coarse_history:],
                    ]
                )

            alternatingLogitsProcessor = AlternatingCodebooksLogitsProcessor(
                input_coarse.shape[1],
                semantic_generation_config.semantic_vocab_size,
                codebook_size,
            )

            output_coarse = super().generate(
                input_coarse,
                logits_processor=[alternatingLogitsProcessor],
                max_new_tokens=min(sliding_window_len, max_generated_len - total_generated_len),
                generation_config=coarse_generation_config,
                **kwargs,
            )

            input_coarse_len = input_coarse.shape[1]

            x_coarse = ops.hstack([x_coarse, output_coarse[:, input_coarse_len:]])
            total_generated_len = x_coarse.shape[1] - len_coarse_history

            del output_coarse

        coarse_output = x_coarse[:, len_coarse_history:]

        if return_output_lengths:
            return coarse_output, output_lengths

        return coarse_output


class BarkFineModel(BarkPreTrainedModel):

    """
    BarkFineModel is a model for generating fine acoustics tokens from input coarse acoustics tokens and optional prompts,
    building on the BarkPreTrainedModel base class.

    This class provides methods for resizing token embeddings, tying weights between input and output embeddings, and
    generating fine acoustics tokens based on input coarse acoustics tokens and generation configurations.

    Attributes:
        config: Configuration object containing model settings.

    Methods:
        resize_token_embeddings:
            Resizes the input token embeddings matrix of the model, taking care of tying weights embeddings afterwards
            if necessary.

        tie_weights():
            Ties the weights between the input embeddings list and the output embeddings list.

        generate:
            Generates fine acoustics tokens from input coarse acoustics tokens and optional speaker prompts,
            following specified generation configurations.

        _resize_token_embeddings:
            Helper method to resize the token embeddings matrix.

        get_input_embeddings() -> nn.CellList:
            Returns the input embeddings layers.

        set_input_embeddings(new_embeddings):
            Sets new input embeddings layers.

        get_output_embeddings() -> nn.CellList:
            Returns the output embeddings layers.

        set_output_embeddings(new_output_embeddings):
            Sets new output embeddings layers.

        construct:
            Constructs the model for a specific codebook index, handling input tokens, masks, and labels accordingly.
    """
    base_model_prefix = "fine_acoustics"
    config_class = BarkFineConfig
    main_input_name = "codebook_idx"

    def __init__(self, config):
        """
        Initializes a BarkFineModel object.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.
            config (Config):
                An object containing configuration parameters for the model.
                Parameters:

                - input_vocab_size (int): The size of the input vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - block_size (int): The size of the blocks in the model.
                - dropout (float): The dropout rate.
                - num_layers (int): The number of layers in the model.
                - output_vocab_size (int): The size of the output vocabulary.
                - n_codes_total (int): The total number of codes used.
                - n_codes_given (int): The number of codes given.

        Returns:
            None.

        Raises:
            None.
        """
        # non-causal gpt-like model with one embedding layer and one lm_head for each codebook of Encodec
        super().__init__(config)
        self.config = config

        # initialize a modified non causal GPT-like model
        # note that for there is one embedding layer and one lm_head for each codebook of Encodec
        self.input_embeds_layers = nn.CellList(
            [nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)]
        )
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        self.drop = nn.Dropout(p=config.dropout)

        self.layers = nn.CellList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])

        self.layernorm_final = nn.LayerNorm(config.hidden_size)

        self.lm_heads = nn.CellList(
            [
                nn.Dense(config.hidden_size, config.output_vocab_size, has_bias=False)
                for _ in range(config.n_codes_given, config.n_codes_total)
            ]
        )
        self.gradient_checkpointing = False
        self.n_codes_total = config.n_codes_total

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the BarkFineModel.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.

        Returns:
            None: This method returns the input embeddings layers for the BarkFineModel.

        Raises:
            None.
        """
        # one embedding layers for each codebook
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the BarkFineModel.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.
            new_embeddings (object): The new embeddings to set for the input_embeds_layers attribute.

        Returns:
            None.

        Raises:
            None.
        """
        # one embedding layers for each codebook
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        """
        This method is defined in the class 'BarkFineModel' and is used to retrieve the output embeddings of the model.

        Args:
            self: An instance of the 'BarkFineModel' class.

        Returns:
            None.

        Raises:
            None.
        """
        # one lm_head for each codebook
        return self.lm_heads

    def set_output_embeddings(self, new_output_embeddings):
        """
        Method to set new output embeddings for the BarkFineModel.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.
            new_output_embeddings (object): New output embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        # one lm_head for each codebook
        self.lm_heads = new_output_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        """
        Resize the token embeddings for the BarkFineModel.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.
            new_num_tokens (int): The new number of tokens to resize the embeddings to.
            pad_to_multiple_of (int or None): If provided, the embeddings will be padded to be a multiple of this value.

        Returns:
            None: The method updates the token embeddings of the model in place.

        Raises:
            TypeError: If new_num_tokens is not an integer.
            ValueError: If new_num_tokens is less than or equal to 0.
            ValueError: If pad_to_multiple_of is provided but is not an integer.
            ValueError: If pad_to_multiple_of is less than or equal to 0.
        """
        old_embeddings_list = self.get_input_embeddings()
        new_embeddings_list = nn.CellList(
            [
                self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
                for old_embeddings in old_embeddings_list
            ]
        )
        self.set_input_embeddings(new_embeddings_list)
        new_num_tokens = new_embeddings_list[0].weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head_list = self.get_output_embeddings()
            new_lm_head_list = nn.CellList(
                [self._get_resized_lm_head(old_lm_head, new_num_tokens) for old_lm_head in old_lm_head_list]
            )
            self.set_output_embeddings(new_lm_head_list)

        return self.get_input_embeddings()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Returns:
            `nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        self.output_vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def tie_weights(self):
        """
        Tie the weights between the input embeddings list and the output embeddings list.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            self._tied_weights_keys = []
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()

            for i in range(self.config.n_codes_total - self.config.n_codes_given):
                # self.input_embeds_layers[i + 1].weight = self.lm_heads[i].weight
                self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
                self._tied_weights_keys.append(f"lm_heads.{i}.weight")

        for module in self.cells():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def construct(
        self,
        codebook_idx: int,  # an additionnal idx corresponding to the id of the codebook that will be predicted
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        input_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
        """
        Construct and process the input data for the BarkFineModel.

        Args:
            self (BarkFineModel): The instance of the BarkFineModel class.
            codebook_idx (int): Index of the codebook to predict.
            input_ids (Optional[mindspore.Tensor], optional): Input tensor containing the tokenized input sequence. Defaults to None.
            attention_mask (Optional[mindspore.Tensor], optional): Tensor indicating which tokens should be attended to. Defaults to None.
            position_ids (Optional[mindspore.Tensor], optional): Tensor containing the position indices of each input token. Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional): Tensor specifying which attention heads to mask. Defaults to None.
            labels (Optional[mindspore.Tensor], optional): Tensor containing the labels for the masked language modeling task. Defaults to None.
            input_embeds (Optional[mindspore.Tensor], optional): Tensor containing the input embeddings. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dictionary instead of a tuple. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
                If `return_dict` is False, returns a tuple containing the following:

                - None: Placeholder for loss value (None for this method).
                - logits (mindspore.Tensor): Predicted logits for masked language modeling task.
                - all_hidden_states (Tuple[mindspore.Tensor]): Tuple of hidden states for each layer.
                - all_self_attentions (Tuple[mindspore.Tensor]): Tuple of attention weights for each layer.

            If `return_dict` is True, returns a MaskedLMOutput object containing the following attributes:

                - loss (None): Placeholder for loss value (None for this method).
                - logits (mindspore.Tensor): Predicted logits for masked language modeling task.
                - hidden_states (Tuple[mindspore.Tensor]): Tuple of hidden states for each layer.
                - attentions (Tuple[mindspore.Tensor]): Tuple of attention weights for each layer.

        Raises:
            ValueError: If codebook_idx is 0, as it should be predicted by the coarse model.
            ValueError: If both input_ids and input_embeds are specified.
            ValueError: If neither input_ids nor input_embeds are specified.
            ValueError: If batch_size is not defined or less than or equal to 0.
            NotImplementedError: If labels are provided, as training is not implemented yet.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if codebook_idx == 0:
            raise ValueError("Cannot predict 0th codebook - 0th codebook should be predicted by the coarse model")

        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")

        if input_ids is None and input_embeds is None:
            raise ValueError("You have to specify either input_ids or input_embeds")

        if input_ids is not None:
            # the input_embeddings are the sum of the j previous codebooks embeddings before
            # the current codebook_idx codebook

            # forward the GPT model itself
            input_embeds = [
                input_embeds_layer(input_ids[:, :, i]).unsqueeze(-1)
                for i, input_embeds_layer in enumerate(self.input_embeds_layers)
            ]  # token embeddings of shape (b, t, n_embd)
            input_embeds = ops.cat(input_embeds, axis=-1)
            input_embeds = input_embeds[:, :, :, : codebook_idx + 1].sum(axis=-1)

        input_shape = input_embeds.shape[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = ops.arange(0, seq_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)  # shape (1, seq_length)

        position_embeds = self.position_embeds_layer(position_ids)  # position embeddings of shape (1, t, n_embd)

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            # [bsz, to_seq_length] -> [bsz, 1, 1, to_seq_length]
            # from_seq_length is 1 to easily broadcast
            attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.shape[-1],)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.layernorm_final(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_heads[codebook_idx - self.config.n_codes_given](hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            return tuple(v for v in [None, logits, all_hidden_states, all_self_attentions] if v is not None)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def generate(
        self,
        coarse_output: mindspore.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        coarse_generation_config: BarkCoarseGenerationConfig = None,
        fine_generation_config: BarkFineGenerationConfig = None,
        codebook_size: int = 1024,
        history_prompt: Optional[Dict[str, mindspore.Tensor]] = None,
        **kwargs,
    ) -> mindspore.Tensor:
        """
        Generates fine acoustics tokens from input coarse acoustics tokens and an additional optional `Bark` speaker
        prompt.

        Args:
            coarse_output (`mindspore.Tensor` of shape (batch_size, seq_len)):
                Input coarse acoustics ids, i.e the output of `BarkCoarseModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            fine_generation_config (`BarkFineGenerationConfig`):
                Generation config indicating how to generate the fine tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,mindspore.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.

        Returns:
            mindspore.Tensor: Output fine acoustics tokens.
        """
        if semantic_generation_config is None:
            raise ValueError("`semantic_generation_config` has to be provided")

        if coarse_generation_config is None:
            raise ValueError("`coarse_generation_config` has to be provided")

        if fine_generation_config is None:
            raise ValueError("`fine_generation_config` has to be provided")

        # since we don't really use GenerationConfig through the fine model (autoencoder)
        # and since only temperature is used from the classic GenerationConfig parameters
        # manually impose the kwargs priority over the generation config
        temperature = kwargs.get("temperature", fine_generation_config.temperature)

        max_fine_history_length = fine_generation_config.max_fine_history_length
        max_fine_input_length = fine_generation_config.max_fine_input_length

        # shape: (batch, n_coarse_codebooks * seq_len)
        # new_shape: (batch, seq_len, n_coarse_codebooks)
        coarse_output = coarse_output.view(coarse_output.shape[0], -1, coarse_generation_config.n_coarse_codebooks)

        # brings ids into the range [0, codebook_size -1]
        coarse_output = ops.remainder(coarse_output - semantic_generation_config.semantic_vocab_size, codebook_size)
        batch_size = coarse_output.shape[0]

        if history_prompt is not None:
            x_fine_history = ops.repeat_interleave(history_prompt["fine_prompt"].T[None], batch_size, axis=0).astype(mindspore.int64)
            # swapaxes to get to shape (seq_len, n_fine_codebooks)
        else:
            x_fine_history = None

        n_coarse = coarse_generation_config.n_coarse_codebooks

        # pad the last 6th codebooks
        fine_input = ops.pad(
            coarse_output,
            (0, fine_generation_config.n_fine_codebooks - n_coarse),
            "constant",
            codebook_size,
        )

        # prepend history if available (max max_fine_history_length)
        if x_fine_history is not None:
            fine_input = ops.cat([x_fine_history[:, -max_fine_history_length:, :], fine_input], axis=1)

            # len of the fine_history that has been added to fine_input
            n_history = x_fine_history[:, -max_fine_history_length:, :].shape[1]
        else:
            n_history = 0

        n_remove_from_end = 0
        # need to pad if too short (since non-causal model)
        if fine_input.shape[1] < max_fine_input_length:
            n_remove_from_end = max_fine_input_length - fine_input.shape[1]
            fine_input = ops.pad(fine_input, (0, 0, 0, n_remove_from_end), mode="constant", value=codebook_size)

        # we can be lazy about fractional loop and just keep overwriting codebooks.
        # seems that coarse_output.shape[1] - (max_fine_input_length - n_history) is equal to minus n_remove_from_end
        # So if we needed to pad because too short, n_loops is always 1 (because n_remove_from_end > 0)
        # If not, we loop over at least twice.

        n_loops = (coarse_output.shape[1] - (max_fine_input_length - n_history)) / max_fine_history_length
        n_loops = int(np.ceil(n_loops))
        n_loops = max(0, n_loops) + 1

        for n_outer in range(n_loops):
            start_idx = min([n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_input_length])

            start_fill_idx = min(
                [n_history + n_outer * max_fine_history_length, fine_input.shape[1] - max_fine_history_length]
            )
            rel_start_fill_idx = start_fill_idx - start_idx
            input_buffer = fine_input[:, start_idx : start_idx + max_fine_input_length, :]
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                logits = self.construct(n_inner, input_buffer).logits
                if temperature is None or temperature == 1.0:
                    relevant_logits = logits[:, rel_start_fill_idx:, :codebook_size]
                    codebook_preds = ops.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[:, :, :codebook_size] / temperature
                    # apply softmax
                    probs = ops.softmax(relevant_logits, axis=-1)[:, rel_start_fill_idx:max_fine_input_length]
                    # reshape to 2D: (batch_size, seq_len, codebook_size) -> (batch_size*seq_len, codebook_size)
                    probs = probs.reshape((-1, codebook_size))
                    # multinomial then reshape : (batch_size*seq_len)-> (batch_size,seq_len)
                    codebook_preds = ops.multinomial(probs, num_samples=1).view(batch_size, -1)
                codebook_preds = codebook_preds.to(mindspore.int32)
                input_buffer[:, rel_start_fill_idx:, n_inner] = codebook_preds
                del logits, codebook_preds

            # transfer into fine_input
            for n_inner in range(n_coarse, fine_generation_config.n_fine_codebooks):
                fine_input[
                    :, start_fill_idx : start_fill_idx + (max_fine_input_length - rel_start_fill_idx), n_inner
                ] = input_buffer[:, rel_start_fill_idx:, n_inner]
            del input_buffer

        fine_input = fine_input.swapaxes(1, 2)[:, :, n_history:]
        if n_remove_from_end > 0:
            fine_input = fine_input[:, :, :-n_remove_from_end]

        if fine_input.shape[-1] != coarse_output.shape[-2]:
            raise ValueError("input and output should have the same seq_len")

        return fine_input


class BarkModel(BarkPreTrainedModel):
    """
    BarkModel

    This class represents a Bark model that is used for generating audio from an input prompt and an optional speaker prompt. It is a subclass of BarkPreTrainedModel.

    Methods:
        __init__: Initializes the BarkModel instance.
        codec_decode: Turns quantized audio codes into an audio array using the encodec.
        generate:
            Generates audio from an input prompt and an optional speaker prompt.

    Attributes:
        semantic: An instance of BarkSemanticModel.
        coarse_acoustics: An instance of BarkCoarseModel.
        fine_acoustics: An instance of BarkFineModel.
        codec_model: An instance of the AutoModel class.
        config: The configuration object for the BarkModel.

    Example:
        ```python
        >>> from transformers import AutoProcessor, BarkModel
        ...
        >>> processor = AutoProcessor.from_pretrained("suno/bark-small")
        >>> model = BarkModel.from_pretrained("suno/bark-small")
        ...
        >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
        >>> voice_preset = "v2/en_speaker_6"
        ...
        >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)
        ...
        >>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
        >>> audio_array = audio_array.cpu().numpy().squeeze()
        ```

    """
    config_class = BarkConfig

    def __init__(self, config):
        """
        Initializes a new instance of BarkModel.

        Args:
            self (BarkModel): The current instance of the BarkModel class.
            config (dict): A dictionary containing configuration settings for the BarkModel.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.semantic = BarkSemanticModel(config.semantic_config)
        self.coarse_acoustics = BarkCoarseModel(config.coarse_acoustics_config)
        self.fine_acoustics = BarkFineModel(config.fine_acoustics_config)

        self.codec_model = AutoModel.from_config(config.codec_config)

        self.config = config

    def codec_decode(self, fine_output, output_lengths=None):
        """Turn quantized audio codes into audio array using encodec."""
        fine_output = fine_output.swapaxes(0, 1)
        emb = self.codec_model.quantizer.decode(fine_output)

        if output_lengths is not None:
            # encodec uses LSTMs which behaves differently with appended padding
            # decoding with encodec takes around 0.1% of the total generation time
            # to keep generation quality, we break batching
            out = [sample[:, :l].unsqueeze(0) for (sample, l) in zip(emb, output_lengths)]
            audio_arr = [self.codec_model.decoder(sample).squeeze() for sample in out]
        else:
            out = self.codec_model.decoder(emb)
            audio_arr = out.squeeze(1)  # squeeze the codebook dimension

        return audio_arr

    def generate(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        history_prompt: Optional[Dict[str, mindspore.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,
    ) -> mindspore.Tensor:
        """
        Generates audio from an input prompt and an additional optional `Bark` speaker prompt.

        Args:
            input_ids (`Optional[mindspore.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the
                longest generation among the batch.
            history_prompt (`Optional[Dict[str,mindspore.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments. Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *semantic_*, *coarse_*, *fine_* prefix, they will be input for the `generate` method of the
                semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for all sub-models except one.
            return_output_lengths (`bool`, *optional*):
                Whether or not to return the waveform lengths. Useful when batching.

        Returns:
            mindspore.Tensor:
                By default:

                - **audio_waveform** (`mindspore.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.

                When `return_output_lengths=True`:

                - Returns a tuple made of:
                - **audio_waveform** (`mindspore.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.
                - **output_lengths** (`mindspore.Tensor` of shape (batch_size)): The length of each waveform in the batch

        Example:
            ```python
            >>> from transformers import AutoProcessor, BarkModel
            ...
            >>> processor = AutoProcessor.from_pretrained("suno/bark-small")
            >>> model = BarkModel.from_pretrained("suno/bark-small")
            ...
            >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
            >>> voice_preset = "v2/en_speaker_6"
            ...
            >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)
            ...
            >>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
            >>> audio_array = audio_array.cpu().numpy().squeeze()
            ```
        """
        # TODO (joao):workaround until nested generation config is compatible with PreTrained Model
        # todo: dict
        semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
        coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
        fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)

        kwargs_semantic = {
            # if "attention_mask" is set, it should not be passed to CoarseModel and FineModel
            "attention_mask": kwargs.pop("attention_mask", None),
            "min_eos_p": kwargs.pop("min_eos_p", None),
        }
        kwargs_coarse = {}
        kwargs_fine = {}
        for key, value in kwargs.items():
            if key.startswith("semantic_"):
                key = key[len("semantic_") :]
                kwargs_semantic[key] = value
            elif key.startswith("coarse_"):
                key = key[len("coarse_") :]
                kwargs_coarse[key] = value
            elif key.startswith("fine_"):
                key = key[len("fine_") :]
                kwargs_fine[key] = value
            else:
                # If the key is already in a specific config, then it's been set with a
                # submodules specific value and we don't override
                if key not in kwargs_semantic:
                    kwargs_semantic[key] = value
                if key not in kwargs_coarse:
                    kwargs_coarse[key] = value
                if key not in kwargs_fine:
                    kwargs_fine[key] = value

        # 1. Generate from the semantic model
        semantic_output = self.semantic.generate(
            input_ids,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            **kwargs_semantic,
        )

        # 2. Generate from the coarse model
        coarse_output = self.coarse_acoustics.generate(
            semantic_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=self.generation_config.codebook_size,
            return_output_lengths=return_output_lengths,
            **kwargs_coarse,
        )

        output_lengths = None
        if return_output_lengths:
            coarse_output, output_lengths = coarse_output
            # (batch_size, seq_len*coarse_codebooks) -> (batch_size, seq_len)
            output_lengths = output_lengths // coarse_generation_config.n_coarse_codebooks

        # 3. "generate" from the fine model
        output = self.fine_acoustics.generate(
            coarse_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            fine_generation_config=fine_generation_config,
            codebook_size=self.generation_config.codebook_size,
            **kwargs_fine,
        )

        if getattr(self, "fine_acoustics_hook", None) is not None:
            # Manually offload fine_acoustics to CPU
            # and load codec_model to GPU
            # since bark doesn't use codec_model forward pass
            self.fine_acoustics_hook.offload()
            self.codec_model = self.codec_model

        # 4. Decode the output and generate audio array
        audio = self.codec_decode(output, output_lengths)

        if getattr(self, "codec_model_hook", None) is not None:
            # Offload codec_model to CPU
            self.codec_model_hook.offload()

        # if return_output_lengths:
        #     output_lengths = [len(sample) for sample in audio]
        #     audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
        #     return audio, output_lengths

        return audio

__all__ = [
    "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BarkFineModel",
    "BarkSemanticModel",
    "BarkCoarseModel",
    "BarkModel",
    "BarkPreTrainedModel",
    "BarkCausalModel",
]
