# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""MindSpore BLOOM model."""

import math
import warnings
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_bloom import BloomConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"
_CONFIG_FOR_DOC = "BloomConfig"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]


def build_alibi_tensor(attention_mask: mindspore.Tensor, num_heads: int, dtype) -> mindspore.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
        attention_mask (`mindspore.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor

    Returns:
        tensor shaped (batch_size * num_heads, 1, max_seq_len)
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = mindspore.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=mindspore.float32
    )
    powers = ops.arange(1, 1 + closest_power_of_2, dtype=mindspore.int32)
    slopes = ops.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = mindspore.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=mindspore.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = ops.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=mindspore.int32)
        slopes = ops.cat([slopes, ops.pow(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def dropout_add(x: mindspore.Tensor, residual: mindspore.Tensor, prob: float, training: bool) -> mindspore.Tensor:
    """
    Dropout add function

    Args:
        x (`mindspore.tensor`, *required*):
            input tensor
        residual (`mindspore.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = ops.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class BloomAttention(nn.Cell):

    """
    BloomAttention class represents an attention mechanism used in neural network models for processing sequential data.
    This class inherits from nn.Cell and includes methods for initializing the attention mechanism, splitting and merging heads,
    and constructing the attention mechanism for a specific layer. The attention mechanism involves performing operations on
    query, key, and value tensors to compute attention scores and produce context layers. Additionally, it supports features
    such as caching past layers, applying attention masks, and handling head masks. The class also provides options for
    fine-tuning the attention mechanism based on pretraining steps and optimization preferences.
    """
    def __init__(self, config: BloomConfig):
        """
        Initialize the BloomAttention class.

        Args:
            self: The instance of the BloomAttention class.
            config (BloomConfig): 
                An object of type BloomConfig that holds configuration parameters for the attention mechanism.
                
                - pretraining_tp (bool): Indicates whether pretraining is used.
                - slow_but_exact (bool): Indicates if the model focuses on accuracy even if it's slower.
                - hidden_size (int): The size of the hidden layers in the model.
                - n_head (int): The number of attention heads to be used.
                - hidden_dropout (float): The dropout probability for hidden layers.

        Returns:
            None:
                This method initializes various attributes of the BloomAttention instance.

        Raises:
            ValueError: If the `hidden_size` is not divisible by `num_heads`, indicating an incompatible configuration.
        """
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=True)
        self.dense = nn.Dense(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)

    def _split_heads(self, fused_qkv: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`mindspore.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, _ = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`mindspore.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            mindspore.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        residual: mindspore.Tensor,
        alibi: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        Method 'construct' in the class 'BloomAttention'.

        Args:
            self (BloomAttention): The instance of the BloomAttention class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            residual (mindspore.Tensor): The input tensor representing the residual connection.
            alibi (mindspore.Tensor): The tensor used in the computation.
            attention_mask (mindspore.Tensor): The tensor containing attention mask values.
            layer_past (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]): Tuple containing past key and value tensors. Default is None.
            head_mask (Optional[mindspore.Tensor]): The tensor representing head mask. Default is None.
            use_cache (bool): Flag indicating whether to use cache. Default is False.
            output_attentions (bool): Flag indicating whether to output attention scores. Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.swapaxes(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.swapaxes(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = ops.cat((past_key, key_layer), axis=2)
            value_layer = ops.cat((past_value, value_layer), axis=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `mindspore.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == mindspore.float16:
            attention_scores = attention_scores.to(mindspore.float32)
        attn_weights = ops.masked_fill(attention_scores, attention_mask, finfo(attention_scores.dtype, 'min'))
        attention_probs = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = ops.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = ops.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + ops.dense(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Cell):

    """
    BloomMLP is a multi-layer perceptron (MLP) that is used for pre-training in natural language processing (NLP) tasks. 
    This class inherits from nn.Cell and implements the forward propagation logic for the MLP.

    Attributes:
        pretraining_tp (int): The number of times the MLP will be pre-trained.
        slow_but_exact (bool): Flag to indicate if the MLP should be trained slowly for accuracy.
        hidden_dropout (float): The probability of an element being zeroed in the hidden layer during training.

    Methods:
        __init__: Initializes the BloomMLP object with the provided configuration.
        construct: Implements the forward propagation logic for the MLP.

    Example usage:
        ```python
        >>> config = BloomConfig(hidden_size=768, pretraining_tp=3, slow_but_exact=True, hidden_dropout=0.1)
        >>> mlp = BloomMLP(config)
        >>> hidden_states = mindspore.Tensor(shape=(1, 768, 10), dtype=mindspore.float32)
        >>> residual = mindspore.Tensor(shape=(1, 768, 10), dtype=mindspore.float32)
        >>> output = mlp.construct(hidden_states, residual)
        ```

    Note:
        The shapes of hidden_states and residual tensors must match for the forward propagation to work correctly.
    """
    def __init__(self, config: BloomConfig):
        """
        Initializes an instance of the BloomMLP class.

        Args:
            self: The instance of the class.
            config (BloomConfig): The configuration object for the BloomMLP. 
                It specifies the parameters and settings for the model.
                
                - `hidden_size` (int): The size of the hidden layer.
                - `pretraining_tp` (bool): A flag indicating whether the model is in pretraining mode.
                - `slow_but_exact` (bool): A flag indicating whether the model should prioritize accuracy over speed.
                - `hidden_dropout` (float): The dropout rate for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Dense(hidden_size, 4 * hidden_size)
        self.gelu_impl = nn.GELU()
        self.dense_4h_to_h = nn.Dense(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def construct(self, hidden_states: mindspore.Tensor, residual: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output tensor for the BloomMLP model.

        Args:
            hidden_states (mindspore.Tensor): The input tensor representing hidden states.
              This tensor is processed through the model to generate the final output.
            residual (mindspore.Tensor): The residual tensor to be added to the final output.
              This tensor contains additional information to be incorporated in the output.

        Returns:
            mindspore.Tensor: The output tensor generated by the BloomMLP model.
              This tensor represents the final result after processing the hidden states.

        Raises:
            None
        """
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = ops.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + ops.dense(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class BloomBlock(nn.Cell):

    """
    This class represents a block of the Bloom transformer model.
    It contains layers for self-attention and multi-layer perceptron (MLP) operations.

    Attributes:
        input_layernorm (nn.LayerNorm): Layer normalization for the input hidden states.
        num_heads (int): Number of attention heads.
        self_attention (BloomAttention): Self-attention mechanism.
        post_attention_layernorm (nn.LayerNorm): Layer normalization after the self-attention layer.
        mlp (BloomMLP): Multi-layer perceptron for further processing.
        apply_residual_connection_post_layernorm (bool):
            Flag indicating whether to apply residual connection after the post-attention layer normalization.
        hidden_dropout (float): Dropout probability for the hidden states.
    """
    def __init__(self, config: BloomConfig):
        """
        Initializes a BloomBlock object with the provided configuration.

        Args:
            self (BloomBlock): The instance of the BloomBlock class.
            config (BloomConfig):
                An object of type BloomConfig containing configuration parameters for the block.

                - hidden_size (int): The size of the hidden layer.
                - n_head (int): The number of attention heads.
                - layer_norm_epsilon (float): The epsilon value for LayerNorm.
                - apply_residual_connection_post_layernorm (bool):
                Flag indicating whether to apply residual connection post LayerNorm.
                - hidden_dropout (float): The dropout rate for the hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        Constructs the BloomBlock.

        This method applies the BloomBlock layer to the input hidden states and returns the outputs.
        The BloomBlock applies self-attention mechanism followed by a feed-forward neural network.

        Args:
            self (BloomBlock): The instance of the BloomBlock class.
            hidden_states (mindspore.Tensor):
                The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            alibi (mindspore.Tensor): The alibi tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (mindspore.Tensor): The attention mask tensor of shape (batch_size, sequence_length).
            layer_past (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]): Optional.
                The cached key-value pairs of the previous attention layers. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): Optional.
                The mask tensor to nullify selected heads of the self-attention layer. Defaults to None.
            use_cache (bool): Optional. Whether to use cache for the attention layers. Defaults to False.
            output_attentions (bool): Optional. Whether to output the attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, ...]: The outputs of the BloomBlock layer.
                The first element is the output tensor of shape (batch_size, sequence_length, hidden_size),
                followed by other optional outputs if `output_attentions` is True.

        Raises:
            None: This method does not raise any exceptions.
        """
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PreTrainedModel):

    """
    BloomPreTrainedModel is a Python class that extends the functionality of PreTrainedModel.
    It provides methods for initializing weights and converting cache formats to be compatible with the Bloom model.
    The class includes functions for initializing weights based on the type of neural network cell and for
    standardizing or converting cache formats to match specific implementations. Utilize this class to
    facilitate pre-training tasks in NLP models with MindSpore framework.
    """
    config_class = BloomConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BloomBlock"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
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

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]], batch_size: int
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]],
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class BloomModel(BloomPreTrainedModel):

    """
    This class represents a custom implementation of a transformer model called BloomModel.
    It inherits from the BloomPreTrainedModel class and includes functionalities for building the model architecture,
    setting and getting input embeddings, and constructing the model for inference or training.

    Attributes:
        embed_dim (int): The dimension of the word embeddings.
        num_heads (int): The number of attention heads in the model.
        word_embeddings (nn.Embedding): The word embeddings layer.
        word_embeddings_layernorm (nn.LayerNorm): Layer normalization for word embeddings.
        h (nn.CellList): List of BloomBlocks representing the hidden layers of the model.
        ln_f (nn.LayerNorm): Layer normalization for the final hidden states.
        gradient_checkpointing (bool): Flag indicating whether gradient checkpointing is enabled.

    Methods:
        build_alibi_tensor: Builds an alibi tensor for the model.
        get_input_embeddings: Retrieves the current input embeddings.
        set_input_embeddings: Updates the input embeddings with new values.
        construct: Constructs the model for inference or training, handling various input parameters and configurations.

    Note:
        This class is designed for custom transformer-based models and may require specific configurations and input formats.
    """
    def __init__(self, config: BloomConfig):
        """
        Initialize the BloomModel with the provided configuration.

        Args:
            self (BloomModel): The instance of the BloomModel class.
            config (BloomConfig):
                An object containing configuration settings for the BloomModel.

                - BloomConfig should include the following attributes:
                - hidden_size (int): The size of the hidden layer.
                - n_head (int): The number of attention heads.
                - vocab_size (int): The size of the vocabulary.
                - num_hidden_layers (int): The number of hidden layers.
                - layer_norm_epsilon (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.CellList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def build_alibi_tensor(self, attention_mask: mindspore.Tensor, num_heads: int, dtype) -> mindspore.Tensor:
        '''
        This method builds an alibi tensor based on the provided attention_mask, number of heads, and data type.

        Args:
            self (BloomModel): The instance of the BloomModel class.
            attention_mask (mindspore.Tensor): A tensor representing the attention mask.
            num_heads (int): The number of attention heads to use in building the alibi tensor.
            dtype: The data type of the tensor.

        Returns:
            mindspore.Tensor: A tensor representing the built alibi tensor.

        Raises:
            ValueError: If the attention_mask is not a valid mindspore.Tensor.
            TypeError: If the num_heads is not an integer or if the dtype is not a valid data type.
        '''
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the BloomModel.

        Args:
            self: An instance of the BloomModel class.

        Returns:
            Returns the word embeddings of the input tokens.

        Raises:
            None.
        """
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        Sets the input embeddings for the BloomModel class.

        Args:
            self (BloomModel): The instance of the BloomModel class.
            new_embeddings (mindspore.Tensor): The new embeddings to set as input.
                It should be a tensor representing the word embeddings.

        Returns:
            None.

        Raises:
            None.

        This method sets the word_embeddings attribute of the BloomModel instance to the provided new_embeddings.
        The word_embeddings attribute is used as input for the model during forward propagation.
        """
        self.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the BLOOM model based on the input parameters.

        Args:
            self (BloomModel): An instance of the BloomModel class.
            input_ids (Optional[mindspore.Tensor]):
                Input tensor of shape (batch_size, seq_length) containing the input tokens.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]):
                Tuple of length 'n_layer' where each tuple contains two tensors of shape
                (batch_size, num_heads, seq_length, hidden_size//num_heads) representing the past key and value
                respectively. If not provided, initialized with None.
            attention_mask (Optional[mindspore.Tensor]): Input tensor of shape (batch_size, seq_length)
                containing the attention mask values. If None, initialized with ones tensor of shape (batch_size,
                seq_length + past_key_values_length) where past_key_values_length is the length of past_key_values.
                Default: None.
            head_mask (Optional[mindspore.Tensor]): Input tensor of shape (n_layer, num_heads)
                containing the mask values for each head in each layer. If None, initialized with None. Default: None.
            inputs_embeds (Optional[mindspore.Tensor]): Input tensor of shape (batch_size, seq_length, hidden_size)
                containing the embedded input tokens. If None, initialized with the embeddings of input_ids.
                Default: None.
            use_cache (Optional[bool]): Whether to use past_key_values for faster decoding.
                If None, initialized with the value from the model config. Default: None.
            output_attentions (Optional[bool]): Whether to return the attentions tensors of all attention layers.
                If None, initialized with the value from the model config. Default: None.
            output_hidden_states (Optional[bool]): Whether to return the hidden states tensors of all layers.
                If None, initialized with the value from the model config. Default: None.
            return_dict (Optional[bool]): Whether to return a BaseModelOutputWithPastAndCrossAttentions object as
                the output instead of a tuple. If None, initialized with the value from the model config.
                Default: None.

        Returns:
            Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
                A tuple of the following tensors depending on the value of 'return_dict':

                - hidden_states (mindspore.Tensor): Output tensor of shape (batch_size, seq_length, hidden_size)
                containing the output features of the last layer.
                - presents (Tuple[mindspore.Tensor, ...]): Tuple of length 'n_layer' containing tuples of two tensors of
                shape (batch_size, num_heads, seq_length + past_key_values_length,
                hidden_size//num_heads) representing the present key and value respectively.
                - all_hidden_states (Tuple[mindspore.Tensor, ...]): Tuple of length 'n_layer+1' containing the hidden
                states tensors of all layers including the input embeddings. Each tensor has shape
                (batch_size, seq_length, hidden_size).
                - all_self_attentions (Tuple[mindspore.Tensor, ...]): Tuple of length 'n_layer' containing the attention
                tensors of all attention layers. Each tensor has shape (batch_size, num_heads,
                seq_length + past_key_values_length, seq_length + past_key_values_length).

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or neither of them are provided,
                or if there are any unexpected arguments passed in.
            FutureWarning: If position_ids argument is provided (now deprecated), a warning is issued indicating that
                it has no functionality in BLOOM and will be removed in v5.0.0.
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `mindspore.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length_with_past))

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        causal_mask = causal_mask.bool()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomForCausalLM(BloomPreTrainedModel):

    """
    The `BloomForCausalLM` class is a subclass of `BloomPreTrainedModel` and represents a model
    for causal language modeling using the BLOOM architecture.

    Causal language modeling is the task of predicting the next token in a sequence given the previous tokens.
    The BLOOM architecture is specifically designed for this task, utilizing a transformer model with an additional language modeling head.

    The class has the following methods:

    - `__init__`: Initializes the `BloomForCausalLM` instance with a configuration object.
    - `get_output_embeddings`: Returns the language modeling head.
    - `set_output_embeddings`: Sets the language modeling head to the provided embeddings.
    - `prepare_inputs_for_generation`: Prepares the inputs for generation by removing the prefix length from the
    input sequence and converting the past key values to BLOOM cache format.
    - `construct`: Constructs the BLOOM model by passing the inputs through the transformer and language modeling head.
    Optionally computes the loss if labels are provided.
    - `_reorder_cache`: Reorders the past key values cache to match the beam indices during beam search or beam sampling.

    Additionally, the class inherits all the properties and methods from the `BloomPreTrainedModel` class.

    Note:
        The `labels` parameter in the `construct` method is for language modeling labels, and the `position_ids`
        parameter is deprecated and will be removed in the future.

    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        """
        Initializes a new instance of the BloomForCausalLM class.

        Args:
            self: The current object instance.
            config (BloomConfig): The configuration object containing the model's hyperparameters and settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the BloomForCausalLM model.

        Args:
            self (BloomForCausalLM): The instance of the BloomForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        Sets the output embeddings for the BloomForCausalLM model.

        Args:
            self (BloomForCausalLM): The instance of the BloomForCausalLM class.
            new_embeddings (mindspore.Tensor): The new embeddings to be set for the model's lm_head.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: mindspore.Tensor,
        past_key_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation.

        This method takes 5 parameters: self, input_ids, past_key_values, attention_mask, inputs_embeds. It returns a dictionary containing the model inputs.

        Args:
            self (BloomForCausalLM): The instance of the BloomForCausalLM class.
            input_ids (mindspore.Tensor): The input tensor containing the tokenized input sequence.
            past_key_values (Optional[mindspore.Tensor]): The optional tensor containing the cached key-value pairs from previous generation steps.
            attention_mask (Optional[mindspore.Tensor]): The optional tensor representing the attention mask for the input sequence.
            inputs_embeds (Optional[mindspore.Tensor]): The optional tensor containing the embedded input sequence.

        Returns:
            dict: A dictionary containing the model inputs.
                It may include either 'input_ids' or 'inputs_embeds' depending on the availability of
                'inputs_embeds' and 'past_key_values'.
                It also includes 'past_key_values', 'use_cache', and 'attention_mask' if provided.

        Raises:
            None

        """
        # only last tokens for input_ids if past is not None
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `mindspore.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss = ops.cross_entropy(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        reordered_past = tuple(
            (
                layer_past[0].index_select(0, beam_idx),
                layer_past[1].index_select(0, beam_idx),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)


class BloomForSequenceClassification(BloomPreTrainedModel):

    """
    The 'BloomForSequenceClassification' class represents a fine-tuned sequence classification model based on the Bloom
    architecture. This class inherits from the 'BloomPreTrainedModel' and includes methods
    for model initialization and inference. It provides functionality for computing sequence classification/regression
    loss and handling batch processing. The class also supports different problem types such as
    regression, single-label classification, and multi-label classification.

    The class includes the 'construct' method for generating model outputs and computing loss based on the input data.
    It also handles deprecated arguments and provides warnings for functionality that will be
    removed in future versions. Additionally, the method supports the use of padding tokens and provides appropriate
    error handling for different scenarios.

    The 'BloomForSequenceClassification' class is designed to be used for sequence classification tasks and provides
    flexibility in handling various types of input data and problem types.

    For detailed information on the methods and parameters of this class, please refer to the method docstrings and the class code.
    """
    def __init__(self, config: BloomConfig):
        """
        Initializes an instance of the BloomForSequenceClassification class with the provided configuration.

        Args:
            self: The current instance of the class.
            config (BloomConfig): The configuration object for the BloomForSequenceClassification model.
                It contains various settings and hyperparameters.

                - num_labels (int): The number of labels for the classification task.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Dense(config.hidden_size, config.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `mindspore.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
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
                loss = ops.cross_entropy(pooled_logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(pooled_logits, labels)
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


class BloomForTokenClassification(BloomPreTrainedModel):

    """
    The `BloomForTokenClassification` class is a Python class that represents a model for token classification using
    the BLOOM architecture. This class inherits from the `BloomPreTrainedModel` class.

    Class Attributes:

    - `num_labels`: The number of labels for the token classification task.
    - `transformer`: An instance of the `BloomModel` class that represents the BLOOM transformer model.
    - `dropout`: An instance of the `Dropout` class from the `nn` module for applying dropout regularization.
    - `classifier`: An instance of the `Dense` class from the `nn` module for the final classification layer.

    Methods:
       `__init__`: Initializes a new instance of the `BloomForTokenClassification` class.
            It takes a `BloomConfig` object as input and sets the necessary attributes.
       `construct`: Constructs the BLOOM model for token classification.
            It takes various input tensors and arguments and returns the model output.

            Parameters:

            - `input_ids` (Optional): Tensor containing the input token IDs.
            - `past_key_values` (Optional): Tuple of past key-value tensors.
            - `attention_mask` (Optional): Tensor containing the attention mask.
            - `head_mask` (Optional): Tensor containing the head mask.
            - `inputs_embeds` (Optional): Tensor containing the input embeddings.
            - `labels` (Optional): Tensor containing the labels for computing the loss.
            - `use_cache` (Optional): Boolean indicating whether to use cache.
            - `output_attentions` (Optional): Boolean indicating whether to output attentions.
            - `output_hidden_states` (Optional): Boolean indicating whether to output hidden states.
            - `return_dict` (Optional): Boolean indicating whether to return the output as a `TokenClassifierOutput` object.
            - `**deprecated_arguments`: Deprecated arguments that will be ignored.

            Returns:

            - If `return_dict` is False, returns a tuple containing the logits and other model outputs.
            - If `return_dict` is True, returns a `TokenClassifierOutput` object containing the loss, logits, hidden states, and attentions.
    """
    def __init__(self, config: BloomConfig):
        """
        Initializes an instance of BloomForTokenClassification.

        Args:
            self: The instance of the class.
            config (BloomConfig): The configuration object containing settings for the model.
                It must be an instance of BloomConfig class.
                This parameter is required.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not an instance of BloomConfig.
            AttributeError: If the config object does not contain the required attributes.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = BloomModel(config)
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
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `mindspore.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
            batch_size, seq_length = labels.shape
            loss = ops.cross_entropy(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BloomForQuestionAnswering(BloomPreTrainedModel):

    """
    This class represents a Bloom model for question answering tasks. It is a subclass of BloomPreTrainedModel, which provides the basic structure and functionality for pre-trained models. The
    BloomForQuestionAnswering class includes methods for model construction and inference.

    Attributes:
        transformer: An instance of the BloomModel class, which is responsible for the main transformer
            architecture of the model.
        qa_outputs: A neural network layer that takes the output of the transformer and produces logits
            for start and end positions of the answer span.

    Methods:
        __init__(self, config): Initializes the BloomForQuestionAnswering instance with a given configuration.
        construct(self, input_ids, attention_mask, position_ids, head_mask, inputs_embeds, start_positions,
            end_positions, output_attentions, output_hidden_states, return_dict):
            Constructs the model for question answering based on the given inputs and returns the predicted start
            and end logits of the answer span, as well as other optional outputs.
    """
    def __init__(self, config):
        """
        Initializes the BloomForQuestionAnswering class.
        
        Args:
            self: The object instance.
            config: A dictionary containing the configuration parameters for the model.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
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

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
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
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BloomForCausalLM",
    "BloomModel",
    "BloomPreTrainedModel",
    "BloomForSequenceClassification",
    "BloomForTokenClassification",
    "BloomForQuestionAnswering",
]
