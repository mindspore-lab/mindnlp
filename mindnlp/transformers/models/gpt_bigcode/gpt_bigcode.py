# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""MindNLP gpt_bigcode model"""

import math
from typing import List, Optional, Tuple, Union
from functools import partial
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal
from .gpt_bigcode_config import GPTBigCodeConfig
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

__all__ = ['GPTBigCodeAttention', 'GPTBigCodeMLP', 'GPTBigCodeBlock', 'GPTBigCodePreTrainedModel', 'GPTBigCodeModel',
           'GPTBigCodeForTokenClassification', 'GPTBigCodeForSequenceClassification', 'GPTBigCodeForCausalLM']


def upcast_masked_softmax(
    input_x: mindspore.Tensor, mask: mindspore.Tensor, mask_value: mindspore.Tensor, scale: float, softmax_dtype: mindspore.dtype
):
    """Fuse kernel for upcast masked softmax."""
    input_dtype = input_x.dtype
    input_x = input_x.to(softmax_dtype) * scale
    input_x = ops.where(mask, input_x, mask_value)
    input_x = ops.softmax(input_x, axis=-1).to(input_dtype)
    return input_x


def upcast_softmax(input_x: mindspore.Tensor, scale: float, softmax_dtype: mindspore.dtype):
    """Fuse kernel for upcast softmax."""
    input_dtype = input_x.dtype
    input_x = input_x.to(softmax_dtype) * scale
    input_x = ops.softmax(input_x, axis=-1).to(input_dtype)
    return input_x


def masked_softmax(input_x: mindspore.Tensor, mask: mindspore.Tensor, mask_value: mindspore.Tensor):
    """Fuse kernel for masked softmax."""
    input_x = ops.where(mask, input_x, mask_value)
    input_x = ops.softmax(input_x, axis=-1)
    return input_x


class GPTBigCodeAttention(nn.Cell):
    """GPT BigCode Attention"""
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        """
        Initializes the GPTBigCodeAttention class.
        
        Args:
            self: The instance of the class.
            config: An object containing configuration parameters.
                Must have attributes: multi_query (bool), hidden_size (int), num_attention_heads (int),
                scale_attn_weights (bool), attention_softmax_in_fp32 (bool), scale_attention_softmax_in_fp32 (bool),
                attn_pdrop (float), resid_pdrop (float).
            is_cross_attention: A boolean indicating whether cross-attention is enabled.
            layer_idx: An integer representing the layer index.
        
        Returns:
            None
        
        Raises:
            ValueError: If `embed_dim` is not divisible by num_heads.
            NotImplementedError: If cross-attention is enabled and multi-query attention is not supported.
        """
        super().__init__()
        self.mask_value = None

        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError(
                    "Multi-Query Attention not supported for cross_attention")

            self.c_attn = nn.Dense(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Dense(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = nn.Dense(
                self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        self.c_proj = nn.Dense(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    def _get_mask_value(self, dtype):
        """
        Method _get_mask_value in the class GPTBigCodeAttention.
        
        Args:
            self (object): The instance of the GPTBigCodeAttention class.
            dtype (str): The data type for the mask value. Should be a valid data type.
        
        Returns:
            mask_value: Returns the mask value for the specified data type.
        
        Raises:
            ValueError: If the mask value is None or has a different data type from the specified dtype.
            TypeError: If an invalid data type is provided.
        """
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype:
            tmp_value = np.finfo(mindspore.dtype_to_nptype(dtype)).min
            self.mask_value = ops.full([], Tensor(
                tmp_value, dtype=dtype), dtype=dtype)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        This method calculates and applies attention mechanism to the input query, key, and value tensors in the
        GPTBigCodeAttention class.
        
        Args:
            self: The GPTBigCodeAttention instance.
            query (Tensor): The input query tensor with shape (batch_size, sequence_length, hidden_size)
                if multi_query is False, or (batch_size, sequence_length * num_heads, hidden_size) if multi_query is True.
            key (Tensor): The input key tensor with shape (batch_size * num_heads, hidden_size, sequence_length)
                if multi_query is False, or (batch_size, sequence_length, hidden_size) if multi_query is True.
            value (Tensor): The input value tensor with shape (batch_size * num_heads, sequence_length, hidden_size)
                if multi_query is False, or (batch_size, sequence_length, hidden_size) if multi_query is True.
            attention_mask (Tensor, optional):
                A tensor with shape (batch_size, sequence_length) or
                (batch_size, num_heads, sequence_length, sequence_length) containing attention masks  to be applied to
                the attention scores. Default is None.
            head_mask (Tensor, optional):
                A tensor with shape (batch_size, num_heads, sequence_length) representing the head mask to be applied
                to the attention weights. Default is None.

        Returns:
            Tuple[Tensor, Tensor]:
                A tuple containing the attention output tensor and the attention weights tensor of the specified shapes.

        Raises:
            ValueError: If the shapes of the input tensors are not compatible for the attention calculation.
            TypeError: If the input tensors are not of the expected data type.
            RuntimeError: If an error occurs while performing the attention operation.
        """
        dtype = query.dtype
        softmax_dtype = mindspore.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.shape[-1]
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(
                batch_size, query_length * self.num_heads, self.head_dim)
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(
                batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads,
                              self.head_dim, key_length)

        attn_weights = mindspore.numpy.empty(
            attn_view, dtype=query.dtype)

        attn_weights = ops.zeros_like(attn_weights)
        beta = 1
        attn_weights = Tensor.baddbmm(
            attn_weights, query, key, beta=beta, alpha=scale_factor).view(attn_shape)

        if upcast:
            if attention_mask is None:
                attn_weights = upcast_softmax(
                    attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(softmax_dtype)
                attn_weights = upcast_masked_softmax(
                    attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = ops.where(
                    Tensor(attention_mask, dtype=mindspore.bool_), attn_weights, mask_value)

            attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.swapaxes(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = ops.bmm(attn_weights.view(
                attn_view), value).view(query_shape)
        else:
            attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        layer_past: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[mindspore.Tensor, Optional[mindspore.Tensor]],
        Tuple[mindspore.Tensor, Optional[mindspore.Tensor],
              Tuple[mindspore.Tensor, ...]],
    ]:
        """
        Construct method in the GPTBigCodeAttention class.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states to the attention mechanism.
            layer_past (Optional[mindspore.Tensor]): Past hidden states for the layer. Default is None.
            attention_mask (Optional[mindspore.Tensor]): Mask to prevent attention to certain positions. Default is None.
            head_mask (Optional[mindspore.Tensor]): Mask for individual attention heads. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states from encoder if cross-attention is used.
                Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention. Default is None.
            use_cache (Optional[bool]): Whether to cache the key-value pair for future calls. Default is False.
            output_attentions (Optional[bool]): Whether to output the attention weights. Default is False.

        Returns:
            Union[Tuple[mindspore.Tensor, Optional[mindspore.Tensor]], Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Tuple[mindspore.Tensor, ...]]]:
                Tuple containing the attention output tensor and optionally the present key-value pair and attention
                weights.

        Raises:
            ValueError: If 'q_attn' weights are not defined for cross-attention or if class is not instantiated
                with 'is_cross_attention=True'.
        """
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask.bool()
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split(
                (self.embed_dim, 2 * self.kv_dim), axis=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .swapaxes(1, 2)
                .split((self.head_dim, 2 * self.head_dim), axis=3)
            )

        if layer_past is not None:
            key_value = ops.cat((layer_past, key_value), axis=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), axis=-1)

        attn_output, attn_weights = self._attn(
            query, key.swapaxes(-1, -2), value, attention_mask, head_mask)

        if not self.multi_query:
            attn_output = attn_output.swapaxes(
                1, 2).reshape(hidden_states.shape)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.swapaxes(1, 2)
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTBigCodeMLP(nn.Cell):
    """GPT BigCode MLP"""
    def __init__(self, intermediate_size, config):
        """
        Initializes an instance of the GPTBigCodeMLP class.

        Args:
            self: The object itself.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object with various settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Dense(embed_dim, intermediate_size)
        self.c_proj = nn.Dense(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: Optional[Tuple[mindspore.Tensor]]) -> mindspore.Tensor:
        """
        This method constructs a multi-layer perceptron for the GPT (Generative Pretrained Transformer) model
        using the provided hidden states.

        Args:
            self: The instance of the GPTBigCodeMLP class.
            hidden_states (Optional[Tuple[mindspore.Tensor]]):
                The hidden states to be processed by the multi-layer perceptron.
                It is an optional tuple of mindspore.Tensor containing the input hidden states.
                If not provided, the method will default to None.

        Returns:
            mindspore.Tensor:
                A tensor representing the processed hidden states after passing through the multi-layer perceptron.

        Raises:
            None
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTBigCodeBlock(nn.Cell):
    """GPT BigCode Block"""
    def __init__(self, config, layer_idx=None):
        """
        Initializes an instance of the GPTBigCodeBlock class.

        Args:
            self: The object instance.
            config (object): An object containing configuration settings for the GPTBigCodeBlock.
            layer_idx (int, optional): The index of the layer. Defaults to None.

        Returns:
            None

        Raises:
            NotImplementedError: If cross-attention is enabled with multi-query architecture (MQA).

        """
        super().__init__()
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(
            [hidden_size], epsilon=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(
            [hidden_size], epsilon=config.layer_norm_epsilon)

        if config.add_cross_attention:
            if config.multi_query:
                raise NotImplementedError(
                    "Cross-attention not implemented for MQA")
            self.crossattention = GPTBigCodeAttention(
                config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, epsilon=config.layer_norm_epsilon)

        self.mlp = GPTBigCodeMLP(self.inner_dim, config)

    def construct(
        self,
        hidden_states: Optional[Tuple[mindspore.Tensor]],
        layer_past: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[mindspore.Tensor], Tuple[mindspore.Tensor,
                                       mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]
    ]:
        """
        This method constructs a GPT (Generative Pre-trained Transformer) big code block.

        Args:
            self: The instance of the class.
            hidden_states (Optional[Tuple[mindspore.Tensor]]): The input hidden states.
            layer_past (Optional[mindspore.Tensor]): The past hidden states of the layer.
            attention_mask (Optional[mindspore.Tensor]): The attention mask to mask some positions in the input.
            head_mask (Optional[mindspore.Tensor]): The mask applied to the heads of the multi-head attention.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder.
            use_cache (Optional[bool]): Flag to indicate whether to use cache for faster decoding.
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions.

        Returns:
            Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]]:
                The output of the method which may include the hidden states and optionally attention scores.

        Raises:
            ValueError:
                If `encoder_hidden_states` are passed but the cross-attention layers are not instantiated with the
                flag `config.add_cross_attention=True`.
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

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attn_outputs[2:]

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


class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTBigCodeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBigCodeBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, (GPTBigCodeMLP, GPTBigCodeAttention)):
            sigma = self.config.initializer_range / math.sqrt(2 * self.config.n_layer)
            cell.c_proj.weight.set_data(initializer(Normal(sigma=sigma),cell.c_proj.weight.shape, cell.c_proj.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=self.config.initializer_range),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer(
                    'zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(sigma=self.config.initializer_range),
                                          cell.weight.shape,
                                          cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer(
                'ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(
                'zeros', cell.bias.shape, cell.bias.dtype))

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Set the gradient checkpointing for a given module in the GPTBigCodePreTrainedModel.

        Args:
            self (GPTBigCodePreTrainedModel): The instance of the GPTBigCodePreTrainedModel.
            module (object): The module for which the gradient checkpointing needs to be set.
            value (bool): The boolean value indicating whether to enable gradient checkpointing for the module.

        Returns:
            None.

        Raises:
            TypeError: If the module is not an instance of GPTBigCodeModel.
        """
        if isinstance(module, GPTBigCodeModel):
            module.gradient_checkpointing = value

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Sets the gradient checkpointing flag for a specific module in a GPTBigCodePreTrainedModel instance.

        Args:
            self (GPTBigCodePreTrainedModel): The GPTBigCodePreTrainedModel instance.
            module (GPTBigCodeModel): The module for which to set the gradient checkpointing flag.
            value (bool): The value to set the gradient checkpointing flag to. Default is False.

        Returns:
            None.

        Raises:
            None.

        Note:
            Gradient checkpointing is a memory optimization technique used during training of deep neural networks.
            When the gradient checkpointing flag is set to True for a specific module, intermediate activations are not
            stored during the forward pass, which reduces the memory usage at the cost of recomputing those activations
            during the backward pass. This can be useful for models with large memory requirements.
        """
        if isinstance(module, GPTBigCodeModel):
            module.gradient_checkpointing = value

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

class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    """GPT BigCode Model"""
    def __init__(self, config):
        """
        __init__

        Initializes the GPTBigCodeModel class.

        Args:
            self(GPTBigCodeModel):
                The instance of the GPTBigCodeModel class.
            config(Config):  An instance of the Config class containing configuration parameters for the model.
                The configuration parameters include:

                - multi_query: bool
                Specifies if the model supports multiple queries.
                - hidden_size: int
                Specifies the dimension of the hidden layers.
                - vocab_size: int
                Specifies the size of the vocabulary.
                - max_position_embeddings: int
                Specifies the maximum number of positions for embeddings.
                - embd_pdrop: float
                Specifies the dropout probability for the embeddings.
                - num_hidden_layers: int
                Specifies the number of hidden layers in the model.
                - layer_norm_epsilon: float
                Specifies the epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([GPTBigCodeBlock(config, layer_idx=i)
                              for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(
            [self.embed_dim], epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        max_positions = config.max_position_embeddings
        self.bias = Tensor(
            np.tril(np.ones((max_positions, max_positions))), mindspore.bool_)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the GPTBigCodeModel.

        Args:
            self: The instance of the GPTBigCodeModel class.

        Returns:
            None: This method returns the input embeddings which are of type None.

        Raises:
            None.
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the GPTBigCodeModel.

        Args:
            self (GPTBigCodeModel): The instance of the GPTBigCodeModel class.
            new_embeddings (object): The new input embeddings to be set for the model.
                It can be of any valid type.

        Returns:
            None.

        Raises:
            None.
        """
        self.wte = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the GPTBigCodeModel.

        Args:
            self (GPTBigCodeModel): The instance of the GPTBigCodeModel class.
            input_ids (Optional[mindspore.Tensor], optional): The input sequence tensor. Defaults to None.
            past_key_values (Optional[List[mindspore.Tensor]], optional):
                List of tensors containing the past key values of the model. Defaults to None.
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor. Defaults to None.
            token_type_ids (Optional[mindspore.Tensor], optional): The token type ids tensor. Defaults to None.
            position_ids (Optional[mindspore.Tensor], optional): The position ids tensor. Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional): The head mask tensor. Defaults to None.
            inputs_embeds (Optional[mindspore.Tensor], optional): The input embeddings tensor. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor], optional): The hidden states of the encoder.
                Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor], optional): The attention mask for the encoder.
                Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: The output of the GPTBigCodeModel.
                Returns a tuple or a BaseModelOutputWithPastAndCrossAttentions object depending on the value of
                return_dict.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If batch_size is less than or equal to 0.
            AssertionError: If the encoder_attention_mask has an invalid dimension.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(
                input_ids, attention_mask)
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0].shape[-2]

        if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:,
                                            past_length: input_shape[-1] + past_length:]
        elif position_ids is None:
            position_ids = ops.arange(
                past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)

        # Self-attention mask.
        query_length = input_shape[-1]
        key_length = past_length + query_length
        self_attention_mask = self.bias[None, key_length - query_length: key_length, :key_length]

        if attention_mask is not None:
            self_attention_mask = self_attention_mask * \
                attention_mask.view(batch_size, 1, -1)
            self_attention_mask = self_attention_mask.bool()

        # MQA models: (batch_size, query_length, n_heads, key_length)
        # MHA models: (batch_size, n_heads, query_length, key_length)
        attention_mask = ops.unsqueeze(
            self_attention_mask, 2 if self.multi_query else 1)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if (
            self.config.add_cross_attention
            and encoder_hidden_states is not None
            and encoder_attention_mask is not None
        ):
            if encoder_attention_mask.dim() == 2:
                encoder_attention_mask.unsqueeze(1)
            assert encoder_attention_mask.dim() == 3
            encoder_attention_mask = encoder_attention_mask.bool(
            ).unsqueeze(2 if self.multi_query else 1)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.shape[-1],)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions = all_self_attentions + \
                    (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    """GPT BigCode for CausalLM"""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the GPTBigCodeForCausalLM class.

        Args:
            self (object): The instance of the class.
            config (object): A configuration object containing settings for the GPTBigCodeForCausalLM model.
                It should include parameters such as n_embd (embedding dimension) and vocab_size (vocabulary size).

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Dense(
            config.n_embd, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the GPTBigCodeForCausalLM model.

        Args:
            self (GPTBigCodeForCausalLM): The instance of the GPTBigCodeForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the GPTBigCodeForCausalLM model.

        Args:
            self (GPTBigCodeForCausalLM): The instance of the GPTBigCodeForCausalLM class.
            new_embeddings (object): The new embeddings to be set as output embeddings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        '''
        Prepare inputs for generation.

        Args:
            self (GPTBigCodeForCausalLM): An instance of the GPTBigCodeForCausalLM class.
            input_ids (torch.Tensor): The input tensor of shape [batch_size, sequence_length].
            past_key_values (tuple, optional): The tuple of past key values. Default is None.
            inputs_embeds (torch.Tensor, optional):
                The embedded inputs tensor of shape [batch_size, sequence_length, embedding_size]. Default is None.

        Returns:
            dict: A dictionary containing the model inputs for generation.
                The dictionary may contain the following keys:

                - 'inputs_embeds' (torch.Tensor): The embedded inputs tensor.
                - 'input_ids' (torch.Tensor): The input tensor.
                - 'past_key_values' (tuple): The tuple of past key values.
                - 'use_cache' (bool): Whether to use cache.
                - 'position_ids' (torch.Tensor): The position ids tensor.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'token_type_ids' (torch.Tensor): The token type ids tensor.

        Raises:
            None.
        '''
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1).to(mindspore.int32))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
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
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past_key_values)

class GPTBigCodeForSequenceClassification(GPTBigCodePreTrainedModel):
    """GPT BigCode for Sequence Classification"""
    def __init__(self, config):
        """
        Initializes a new instance of the GPTBigCodeForSequenceClassification class.

        Args:
            self: The object itself.
            config (GPTBigCodeConfig): The configuration object specifying the model's hyperparameters and settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTBigCodeModel(config)
        self.score = nn.Dense(config.n_embd, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.ne(
                    input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

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
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPTBigCodeForTokenClassification(GPTBigCodePreTrainedModel):
    """GPT BigCode for Token Classification"""
    def __init__(self, config):
        """
        Initializes an instance of the GPTBigCodeForTokenClassification class.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the model.
                It must have the following attributes:

                - num_labels: An integer specifying the number of output labels.
                - classifier_dropout: (optional) A float specifying the dropout rate for the classifier layer.
                - hidden_dropout: (optional) A float specifying the dropout rate for hidden layers.

                Note:
                    If both classifier_dropout and hidden_dropout are provided, classifier_dropout takes precedence.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPTBigCodeModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
