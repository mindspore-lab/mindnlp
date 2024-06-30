# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
""" MindNLP GPT Neo model."""

import os
from typing import Union, Optional, Tuple
from functools import partial
import numpy as np
import mindspore
from mindspore import ops, nn, Parameter, Tensor, dtype_to_nptype
from mindspore.common.initializer import initializer, Normal
from mindnlp.utils import logging

from ...modeling_utils import PreTrainedModel
from .gpt_neo_config import GPTNeoConfig
from ...activations import ACT2FN


logger = logging.get_logger(__name__)

class GPTNeoSelfAttention(nn.Cell):
    """
    GPTNeo SelfAttention.
    """
    def __init__(self, config, attention_type):
        """
        Initializes the GPTNeoSelfAttention class.
        
        Args:
            self: The object instance itself.
            config: A configuration object containing various settings for the attention mechanism.
                It must have a 'max_position_embeddings', 'window_size', 'attention_dropout', 'resid_dropout',
                'hidden_size', and 'num_heads' attribute.
            attention_type: A string specifying the type of attention mechanism to use. 
                Must be either 'local' or 'global'.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the embed_dim is not divisible by num_heads.
        """
        super().__init__()

        max_positions = config.max_position_embeddings
        bias = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.bool_)).view(
            1, 1, max_positions, max_positions
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = ops.bitwise_xor(bias, ops.tril(
                bias, -config.window_size)).astype(mindspore.bool_)

        self.bias = Parameter(bias, requires_grad=False)
        self.masked_bias = Parameter(Tensor(-1e9), requires_grad=False)

        self.attn_dropout = nn.Dropout(p=float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(p=float(config.resid_dropout))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        # (batch, head, seq_length, head_features)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        This method calculates the attention weights and output for GPT-Neo self-attention mechanism.
        
        Args:
            self (GPTNeoSelfAttention): The instance of the GPTNeoSelfAttention class.
            query (Tensor): The input tensor representing the query. 
                It should be of shape (batch_size, sequence_length, hidden_size).
            key (Tensor): The input tensor representing the key. 
                It should be of shape (batch_size, sequence_length, hidden_size).
            value (Tensor): The input tensor representing the value. 
                It should be of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): 
                An optional tensor representing the attention mask. 
                It should be of shape (batch_size, 1, sequence_length, sequence_length) and defaults to None.
            head_mask (Tensor, optional): An optional tensor representing the head mask. 
                It should be of shape (num_heads, sequence_length, sequence_length) and defaults to None.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the attention output tensor and the attention weights tensor.
            
        Raises:
            ValueError: If the shapes of query, key, or value are not compatible for matrix multiplication.
            ValueError: If the shapes of attention_mask or head_mask are not compatible with the expected shapes.
            TypeError: If the input tensors are not of type 'Tensor'.
            RuntimeError: If any runtime error occurs during the computation.
        """
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.astype(mindspore.float32)
        key = key.astype(mindspore.float32)

        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length -
                                query_length: key_length, :key_length]
        mask_value = Tensor(np.finfo(dtype_to_nptype(attn_weights.dtype)).min)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        mask_value = Tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = ops.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        This method constructs the self-attention mechanism for the GPTNeo model.
        
        Args:
            self: The GPTNeoSelfAttention instance.
            hidden_states (tensor): The input hidden states for the attention mechanism.
            attention_mask (tensor, optional): Mask to avoid attending to specific positions. Defaults to None.
            layer_past (tuple, optional): Tuple containing the past key and value tensors. Defaults to None.
            head_mask (tensor, optional): Mask to nullify specific heads of the attention mechanism. Defaults to None.
            use_cache (bool, optional): Flag indicating whether to use the cache for fast decoding. Defaults to False.
            output_attentions (bool, optional): Flag indicating whether to output the attention weights. Defaults to False.
        
        Returns:
            tuple: A tuple containing the attention output tensor and the present state tuple. 
                The present state tuple is of type 'tuple' and contains the current key and value tensors
                if use_cache is True, otherwise it is None.
        
        Raises:
            None.
        """
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTNeoAttention(nn.Cell):
    """
    GPTNEO Attention.
    """
    def __init__(self, config, layer_id=0):
        """
        Initialize the GPTNeoAttention class.
        
        Args:
            self: The instance of the class.
            config: The configuration object containing the attention layer settings.
            layer_id (int, optional): The ID of the attention layer. Defaults to 0.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: If the attention type specified in the config is not 'global' or 'local'.
            TypeError: If the attention type is not a string.
        """
        super().__init__()
        self.layer_id = layer_id
        self.attention_layers = config.attention_layers
        self.attention_type = self.attention_layers[layer_id]

        if self.attention_type in ["global", "local"]:
            self.attention = GPTNeoSelfAttention(config, self.attention_type)
        else:
            raise NotImplementedError(
                "Only attn layer types 'global' and 'local' exist, but got `config.attention_layers`: "
                f"{config.attention_layers}. Select attn layer types from ['global', 'local'] only."
            )

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the attention mechanism for the GPTNeoAttention class.
        
        Args:
            self: The instance of the GPTNeoAttention class.
            hidden_states (tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            layer_past (tuple, optional):
                The past states of shape (batch_size, num_heads, past_sequence_length, head_size) to be used for
                attention computation. Defaults to None.
            attention_mask (tensor, optional):
                The attention mask tensor of shape (batch_size, sequence_length) or
                (batch_size, num_heads, sequence_length, sequence_length) to mask attention scores. Defaults to None.
            head_mask (tensor, optional):
                The head mask tensor of shape (num_heads,) or (batch_size, num_heads) to mask attention heads.
                Defaults to None.
            use_cache (bool, optional): Whether to use cached states for attention computation. Defaults to False.
            output_attentions (bool, optional): Whether to output the attention weights. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPTNeoMLP(nn.Cell):
    """
    GPTNeo MLP.
    """
    # in MLP: intermediate_size= 4 * hidden_size
    def __init__(self, intermediate_size, config):
        """
        Initializes the GPTNeoMLP class.

        Args:
            self: The instance of the class.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object containing parameters for the method.

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
        self.dropout = nn.Dropout(p=float(config.resid_dropout))

    def construct(self, hidden_states):
        """
        Constructs the multi-layer perceptron (MLP) component of the GPT-Neo model.

        Args:
            self (GPTNeoMLP): An instance of the GPTNeoMLP class.
            hidden_states (tensor): The input hidden states to be processed by the MLP.

        Returns:
            hidden_states: The processed hidden states after passing through the MLP layers.

        Raises:
            None.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTNeoBlock(nn.Cell):
    """
    GPTNeo Block.
    """
    def __init__(self, config, layer_id):
        """
        Initializes a GPTNeoBlock instance.

        Args:
            self: The object itself.
            config (Config): The configuration object containing various settings.
            layer_id (int): The ID of the layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(
            (hidden_size,), epsilon=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(
            (hidden_size,), epsilon=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Method 'construct' in the class 'GPTNeoBlock' constructs the output of a GPTNeo block.

        Args:
            self (class object): The instance of the class.
            hidden_states (tensor): The input hidden states to the block.
            layer_past (tensor, optional): The past hidden states from previous layers. Default is None.
            attention_mask (tensor, optional): Mask to prevent attention to certain positions. Default is None.
            head_mask (tensor, optional): Mask to prevent attention to certain heads. Default is None.
            use_cache (bool): If True, cache the outputs for faster decoding. Default is False.
            output_attentions (bool): If True, outputs attention weights. Default is False.

        Returns:
            outputs: (tuple) A tuple containing the updated hidden states and additional outputs from the block.

        Raises:
            None
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
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

        # hidden_states, present, (attentions, cross_attentions)
        return outputs


class GPTNeoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTNeoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoBlock"]

    def init_model_weights(self):
        """
        initialize model weights.
        """
    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, (nn.Dense,)):
            cell.weight.set_data(initializer(Normal(
                sigma=self.config.initializer_range, mean=0.0)), cell.weight.shape, cell.weight.dtype)
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros'),
                                     cell.bias.shape, cell.bias.dtype)
        elif isinstance(cell, nn.Embedding):
            cell.weight.set_data(initializer(Normal(
                sigma=self.config.initializer_range, mean=0.0)), cell.weight.shape, cell.weight.dtype)
            if cell.padding_idx is not None:
                zeroslike = ops.ZerosLike()
                cell.weight.data[cell.padding_idx] = zeroslike(
                    cell.weight.data[cell.padding_idx])
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.set_data(initializer('zeros'),
                               cell.bias.shape, cell.bias.dtype)
            cell.weight.data = ops.fill(
                cell.weight.data.dtype, cell.weight.data.shape, 1.0)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.
        """
    def set_input_embeddings(self, new_embeddings: "nn.Cell"):
        """
        Set model's input embeddings.
        """
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """
    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
    def save(self, save_dir: Union[str, os.PathLike]):
        "save pretrain model"
    def _set_gradient_checkpointing(self, module, value=False):
        """
        Sets the gradient checkpointing flag for the specified module in a GPTNeoPreTrainedModel.

        Args:
            self (GPTNeoPreTrainedModel): The instance of the GPTNeoPreTrainedModel.
            module (object): The module for which gradient checkpointing will be set.
            value (bool): The boolean value indicating whether to enable gradient checkpointing for the module.

        Returns:
            None.

        Raises:
            TypeError: If the provided module is not an instance of GPTNeoModel.
        """
        if isinstance(module, GPTNeoModel):
            module.gradient_checkpointing = value

    # TODO
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
    def _backward_compatibility_gradient_checkpointing(self):
        """
        Support gradient_checkpointing.
        """
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))


class GPTNeoModel(GPTNeoPreTrainedModel):
    """
    GPTNeo Model
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoModel class.

        Args:
            self: The GPTNeoModel instance.
            config:
                An object containing the configuration settings for the model. It should have the following attributes:

                - hidden_size (int): The dimensionality of the hidden states and embeddings.
                - vocab_size (int): The size of the vocabulary.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - embed_dropout (float): The dropout probability for the embeddings.
                - num_layers (int): The number of layers in the model.
                - layer_norm_epsilon (float): The epsilon value used in layer normalization.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(p=float(config.embed_dropout))
        self.h = nn.CellList([GPTNeoBlock(config, layer_id=i)
                              for i in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(
            (self.embed_dim,), epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        return the input embeddings layer
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        set the input embeddings layer
        """
        self.wte = new_embeddings

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        '''
        Constructs the GPTNeoModel.

        Args:
            self: The GPTNeoModel instance.
            input_ids (Optional[Tensor]): The input token IDs. Default: None.
            past_key_values (Optional[Tuple[Tensor]]): Cached key-value states from previous forward passes. Default: None.
            attention_mask (Optional[Tensor]): The attention mask. Default: None.
            token_type_ids (Optional[Tensor]): The token type IDs. Default: None.
            position_ids (Optional[Tensor]): The position IDs. Default: None.
            head_mask (Optional[Tensor]): The head mask. Default: None.
            inputs_embeds (Optional[Tensor]): The embedded inputs. Default: None.
            use_cache (Optional[bool]): Whether to use cache for faster decoding. Default: None.
            output_attentions (Optional[bool]): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default: None.

        Returns:
            Tuple[Tensor]:
                A tuple containing the hidden states, cached key-value states, all hidden states (if enabled),
                and all self-attentions (if enabled).

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If batch_size is not defined or <= 0.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = ops.arange(
                past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
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
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.astype(
                dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * \
                (Tensor(np.finfo(dtype_to_nptype(self.dtype)).min))

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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # TODO
            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, use_cache, output_attentions)

            #         return custom_forward

            #     outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #         hidden_states,
            #         None,
            #         attention_mask,
            #         head_mask[i],
            #     )
            # else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + \
                    (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)


class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    """
    GPTNeo For CausalLM.
    """
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the GPTNeoForCausalLM class.

        Args:
            self: The object instance.
            config: An instance of the configuration class that holds the model configuration settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Dense(
            config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        return the output embedding layers.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        set the output embedding layers.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        prepare inputs for generation.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
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
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.astype(mindspore.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            lm_logits = lm_logits.astype(hidden_states.dtype)
            loss = loss.astype(hidden_states.dtype)

        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[Tensor]], beam_idx: Tensor
    ) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx)
                  for past_state in layer_past)
            for layer_past in past_key_values
        )


class GPTNeoForSequenceClassification(GPTNeoPreTrainedModel):
    """
    GPTNeo For Sequence Classification.
    """
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the GPTNeoForSequenceClassification class.

        Args:
            self: The instance of the GPTNeoForSequenceClassification class.
            config: An instance of the configuration class containing the model configuration parameters.
                It must have the attribute 'num_labels' representing the number of labels for sequence classification.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTNeoModel(config)
        self.score = nn.Dense(config.hidden_size,
                              self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    ops.ne(input_ids, self.config.pad_token_id).sum(-1) - 1)
            else:
                sequence_lengths = -1
                logger.warning(
                    "%s will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`",
                    self.__class__.__name__
                )

        pooled_logits = logits[:, sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in {mindspore.int64, mindspore.int32}):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        output = (pooled_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

__all__ = [
        "GPTNeoForCausalLM",
        # "GPTNeoForQuestionAnswering",
        "GPTNeoForSequenceClassification",
        # "GPTNeoForTokenClassification",
        "GPTNeoModel",
        "GPTNeoPreTrainedModel",
    ]
