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
"""MindSpore PanguAlpha GPT2 Model"""

from typing import Tuple
import math

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_gptpangu import GPTPanguConfig


logger = logging.get_logger(__name__)

GPTPangu_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sunzeyeah/pangu-350M-sft",
    "sunzeyeah/pangu-2_6B-sft",
    "sunzeyeah/pangu-350M-reward",
    "sunzeyeah/pangu-350M",
    "sunzeyeah/pangu-2_6B",
    "sunzeyeah/pangu-13B",
]


class GPTPanguAttention(nn.Cell):

    """
    Represents the GPTPanguAttention class, which inherits from nn.Cell.
    This class contains methods for attention mechanism used in GPT (Generative Pre-trained Transformer) models.
    
    Methods:
        __init__: Initializes the GPTPanguAttention instance with the given configuration.
        _attn: Computes the attention mechanism using the query, key, and value tensors, with optional attention and
            head masks.
        _split_heads: Splits the hidden_size dimension of the given tensor into attn_head_size and num_heads.
        _merge_heads: Merges attn_head_size dimension and num_attn_heads dimension into hidden_size.
        construct: Constructs the attention mechanism using the provided hidden_states and optional past layers, masks,
            custom query, cache usage, and attention output flag.
    """
    def __init__(self, config):
        """
        Initializes the GPTPanguAttention class.

        Args:
            self (object): The instance of the class itself.
            config (object):
                An object containing configuration parameters for the attention mechanism.

                - max_position_embeddings (int): The maximum number of positions for positional embeddings.
                - hidden_size (int): The dimension of the hidden state.
                - num_heads (int): The number of attention heads.
                - scale_attn_weights (bool): A flag indicating whether to scale the attention weights.
                - attn_pdrop (float): The dropout probability for attention weights.
                - resid_pdrop (float): The dropout probability for residual connections.

        Returns:
            None.

        Raises:
            ValueError: If the embed_dim is not divisible by num_heads, an exception is raised with a
                detailed error message.
        """
        super().__init__()

        max_positions = config.max_position_embeddings
        self.bias = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.uint8)).view(
                1, 1, max_positions, max_positions
            )
        self.masked_bias = mindspore.tensor(-1e4)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)
        self.c_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        '''
        Method _attn in the GPTPanguAttention class.

        This method calculates the attention weights and applies the attention mechanism to the input values.

        Args:
            self: GPTPanguAttention instance.
                The instance of the GPTPanguAttention class.
            query: tensor, shape [batch_size, num_attention_heads, sequence_length, d_model]
                The query tensor used to calculate the attention scores.
            key: tensor, shape [batch_size, num_attention_heads, sequence_length, d_model]
                The key tensor used to calculate the attention scores.
            value: tensor, shape [batch_size, num_attention_heads, sequence_length, d_model]
                The value tensor which is the input to the attention mechanism.
            attention_mask: tensor, optional
                Mask tensor for the attention scores. If provided, it should have the same shape as attn_weights.
            head_mask: tensor, optional
                Mask tensor for the attention heads. If provided, it should have the same shape as attn_weights.

        Returns:
            attn_output: tensor
                The output tensor after applying the attention mechanism.
                It has the same shape as the input value tensor.
            attn_weights: tensor
                The attention weights representing the importance of each element in the input sequence.

        Raises:
            ValueError: If the dimensions of query, key, or value are not compatible for matrix multiplication.
            TypeError: If any of the input tensors are not of type tensor.
            IndexError: If the dimensions of the input tensors are not as expected for the attention mechanism.
            RuntimeError: If any runtime error occurs during the calculation of attention weights.
        '''
        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)

        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = ops.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        custom_query=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the attention mechanism used in the GPTPangu model.

        Args:
            self (GPTPanguAttention): An instance of the GPTPanguAttention class.
            hidden_states (tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
            layer_past (tuple, optional): A tuple containing the past key and value tensors. Defaults to None.
            attention_mask (tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                Defaults to None.
            head_mask (tensor, optional): The head mask tensor of shape (num_heads, sequence_length, sequence_length).
                Defaults to None.
            custom_query (tensor, optional): The custom query tensor of shape (batch_size, sequence_length, hidden_size).
                Defaults to None.
            use_cache (bool, optional): Whether to use the past key and value tensors. Defaults to False.
            output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.

        Returns:
            tuple:
                A tuple containing the attention output tensor and the present key-value tuple:

                - attn_output (tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
                - present (tuple): A tuple containing the present key and value tensors of shape
                (batch_size, num_heads, sequence_length, head_dim).

        Raises:
            None.
        """
        query = self.q_proj(custom_query) if custom_query is not None else self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTPanguMLP(nn.Cell):

    """
    GPTPanguMLP represents a multi-layer perceptron (MLP) used in the GPT-Pangu model for processing intermediate
    hidden states.

    This class inherits from nn.Cell and contains methods for initializing the MLP layers and processing hidden states
    through a feedforward neural network.

    Attributes:
        c_fc (nn.Dense): Fully connected layer to transform input hidden states.
        c_proj (nn.Dense): Fully connected layer to project intermediate hidden states back to original embed dimension.
        act (ACT2FN[config.activation_function]): Activation function applied to hidden states.
        dropout (nn.Dropout): Dropout layer to add regularization to the model.

    Methods:
        __init__: Initializes the GPTPanguMLP with specified intermediate size and configuration parameters.

        construct: Processes the input 'hidden_states' through the MLP layers and returns the processed hidden states.

    Example:
        ```python
        >>> intermediate_size = 512
        >>> config = Configuration(hidden_size=768, activation_function='gelu', resid_pdrop=0.1)
        >>> mlp = GPTPanguMLP(intermediate_size, config)
        >>> output = mlp.construct(hidden_states)
        ```

    """
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        """
        Initializes the GPTPanguMLP class.

        Args:
            self: The object instance.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object containing hidden_size, activation_function,
                and resid_pdrop attributes.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Dense(embed_dim, intermediate_size)
        self.c_proj = nn.Dense(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states):
        """
        This method constructs the hidden states by applying a series of transformations.

        Args:
            self (GPTPanguMLP): The instance of the GPTPanguMLP class.
            hidden_states (tensor): The input hidden states to be processed.

        Returns:
            None: This method does not return any value explicitly, as the processed hidden states are modified in place.

        Raises:
            None.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTPanguBlock(nn.Cell):

    """
    This class represents a block of the GPTPangu model, containing layers for attention and feed-forward processing.

    Parameters:
        config: An object containing configuration settings for the GPTPanguBlock.

    Attributes:
        ln_1: Layer normalization module for the first layer.
        attn: GPTPanguAttention module for attention processing.
        ln_2: Layer normalization module for the second layer.
        mlp: GPTPanguMLP module for feed-forward processing.

    Methods:
        __init__: Initializes the GPTPanguBlock with the given configuration settings.
        construct:
            Constructs the block by processing the input hidden_states through attention and feed-forward layers.

    Returns:
        outputs:
            A tuple containing the final hidden states after processing.

    Inherits from:
        nn.Cell
    """
    def __init__(self, config):
        """
        Initialize a GPTPanguBlock instance with the provided configuration.

        Args:
            self (GPTPanguBlock): The instance of the GPTPanguBlock class.
            config (GPTPanguConfig):
                The configuration object containing parameters for the block.

                - hidden_size (int): The size of the hidden layers.
                - intermediate_size (int, optional): The size of the intermediate layers. Defaults to None.
                If not provided, it is set to 4 times the hidden size.
                - layer_norm_epsilon (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.attn = GPTPanguAttention(config)
        self.ln_2 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.mlp = GPTPanguMLP(inner_dim, config)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        custom_query=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the GPTPanguBlock.

        Args:
            self: The instance of the class.
            hidden_states (torch.Tensor): The input hidden states of shape `(batch_size, sequence_length, hidden_size)`.
            layer_past (Tuple[torch.Tensor], optional):
                The cached past hidden states of shape `(batch_size, num_heads, sequence_length, hidden_size)`.
                Default is `None`.
            attention_mask (torch.Tensor, optional):
                The attention mask of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
                Default is `None`.
            head_mask (torch.Tensor, optional): The head mask of shape `(num_heads,)`. Default is `None`.
            custom_query (torch.Tensor, optional):
                The custom query tensor of shape `(batch_size, num_heads, sequence_length, hidden_size)`.
                Default is `None`.
            use_cache (bool, optional): Whether to use the cache for the hidden states. Default is `False`.
            output_attentions (bool, optional): Whether to output attentions probabilities. Default is `False`.

        Returns:
            Tuple[torch.Tensor]:
                A tuple containing the following:

                - hidden_states (torch.Tensor):
                The output hidden states of shape `(batch_size, sequence_length, hidden_size)`.
                - layer_past (Tuple[torch.Tensor]):
                The updated cached past hidden states of shape `(batch_size, num_heads, sequence_length, hidden_size)`.
                - attention_weights (List[torch.Tensor], optional):
                The attention weights of shape `(num_layers, num_heads, sequence_length, sequence_length)`,
                only if `output_attentions=True`.
                - other_outputs (List[torch.Tensor], optional):
                Other intermediate outputs, only if `output_attentions=True`.

        Raises:
            None.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            custom_query=custom_query,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTPanguPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTPanguConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense,)):
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

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in cell.parameters_and_names():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.set_data(initializer(Normal(self.config.initializer_range / math.sqrt(2 * self.config.num_layers)),
                                       p.shape, p.dtype))


class GPTPanguModel(GPTPanguPreTrainedModel):

    """GPTPanguModel

    This class represents a GPT-Pangu model, which is a variant of the GPT (Generative Pre-trained Transformer) model.
    It is designed for pre-training and fine-tuning on large-scale Chinese text data. The GPTPanguModel class inherits
    from the GPTPanguPreTrainedModel class.

    Attributes:
        embed_dim (int): The dimensionality of the embedding layer.
        wte (nn.Embedding): The word/token embedding layer.
        wpe (nn.Embedding): The position embedding layer.
        wqe (nn.Embedding): The query embedding layer.
        drop (nn.Dropout): The dropout layer.
        h (nn.CellList): The list of GPTPanguBlock layers.
        ln_f (nn.LayerNorm): The layer normalization layer.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GPTPanguModel class.

        Args:
            self: The instance of the GPTPanguModel class.
            config:
                A configuration object that contains the settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.wqe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([GPTPanguBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve input embeddings from the GPTPanguModel.

        Args:
            self: GPTPanguModel instance. The object instance of the GPTPanguModel class.

        Returns:
           The input embeddings for further processing in the model.

        Raises:
            None.
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the GPTPanguModel.

        Args:
            self (GPTPanguModel): The instance of the GPTPanguModel class.
            new_embeddings: The new input embeddings to be set for the model.
                It should be a tensor or array representing the embeddings.

        Returns:
            None: This method updates the input embeddings of the model in-place.

        Raises:
            TypeError: If the new_embeddings parameter is not of the correct type.
            ValueError: If the new_embeddings parameter is empty or invalid.
        """
        self.wte = new_embeddings

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Constructs the GPTPanguModel.

        Args:
            self (GPTPanguModel): The object instance.
            input_ids (torch.Tensor, optional): The input tensor of shape (batch_size, sequence_length).
                It represents the input token IDs. Defaults to None.
            past_key_values (tuple, optional): The tuple of past key values.
                Each element in the tuple is a tensor of shape (batch_size, num_heads, sequence_length,
                hidden_size//num_heads). Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                It indicates which tokens should be attended to and which ones should not. Defaults to None.
            token_type_ids (torch.Tensor, optional): The token type IDs tensor of shape (batch_size, sequence_length).
                It represents the token type embeddings. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs tensor of shape (batch_size, sequence_length).
                It represents the position embeddings. Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor of shape (num_layers, num_heads).
                It specifies which heads should be masked for each layer. Defaults to None.
            inputs_embeds (torch.Tensor, optional):
                The input embeddings tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the input embeddings directly instead of using input_ids. Defaults to None.
            use_cache (bool, optional): Whether to use cache for faster decoding. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to use a dictionary as the return type. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are provided.
            ValueError: If batch_size is not defined or is less than or equal to 0.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = ops.arange(past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        # top attention custom query
        last_layer_id = len(self.h) - 1
        query_embeds = self.wqe(position_ids)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Final LayerNorm before last query layer
            if i == last_layer_id:
                hidden_states = self.ln_f(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                # custom query
                custom_query=query_embeds if i == last_layer_id else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPTPanguForCausalLM(GPTPanguPreTrainedModel):

    """
    The GPTPanguForCausalLM class represents a Pangu model for causal language modeling.
    It inherits from the GPTPanguPreTrainedModel class.

    This class includes methods for initializing the model, getting and setting output embeddings,
    preparing inputs for generation, and generating outputs based on input data. Additionally, it provides a method
    for re-ordering the past key values cache when using beam search or beam sampling.

    The __init__ method initializes the model with a given configuration and sets up the transformer and lm_head layers.
    The get_output_embeddings and set_output_embeddings methods deal with accessing and  modifying the output embeddings
    for the model. The prepare_inputs_for_generation method prepares input data for generation, considering past key
    values, attention mask, position ids, and token type ids. The construct method constructs outputs based on input data,
    including handling labels for language modeling and computing loss.

    The _reorder_cache method is a static method used to re-order the past_key_values cache when beam search or beam
    sample methods are called, ensuring correct alignment with the beam index at each generation step.
    """
    def __init__(self, config):
        """
        Initializes an instance of the GPTPanguForCausalLM class.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object compatible with the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.transformer = GPTPanguModel(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings of the GPTPanguForCausalLM model.

        Args:
            self: The instance of the GPTPanguForCausalLM class.

        Returns:
            lm_head: This method returns the output embeddings of the model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the GPTPanguForCausalLM model.

        Args:
            self (GPTPanguForCausalLM): The instance of the GPTPanguForCausalLM class.
            new_embeddings (torch.nn.Module): The new embeddings to set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        """
        Prepare inputs for generation.

        Args:
            self (GPTPanguForCausalLM): The instance of the GPTPanguForCausalLM class.
            input_ids (torch.Tensor): The input tensor of token indices representing the sequence.
            past (tuple, optional): The past key values used for fast decoding.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (torch.Tensor): The modified input tensor.
                - 'past_key_values' (tuple): The past key values.
                - 'use_cache' (bool): The flag indicating whether to use cache.
                - 'position_ids' (torch.Tensor): The modified position indices tensor.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'token_type_ids' (torch.Tensor): The modified token type indices tensor.

        Raises:
            None.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.int().cumsum(-1).long() - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
                ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1),
                                     ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[mindspore.Tensor]], beam_idx: mindspore.Tensor) -> Tuple[Tuple[mindspore.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )

__all__ = [
    "GPTPangu_PRETRAINED_MODEL_ARCHIVE_LIST",
    "GPTPanguPreTrainedModel",
    "GPTPanguModel",
    "GPTPanguForCausalLM"
]
