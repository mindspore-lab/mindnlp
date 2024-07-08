# coding=utf-8
# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
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
""" MindSpore CodeGen model."""

from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging, get_default_dtype
from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_codegen import CodeGenConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/codegen-2B-mono"
_CONFIG_FOR_DOC = "CodeGenConfig"


CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-2B-multi",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-6B-multi",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-nl",
    "Salesforce/codegen-16B-multi",
    "Salesforce/codegen-16B-mono",
    # See all CodeGen models at https://hf-mirror.com/models?filter=codegen
]


# Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
def create_sinusoidal_positions(num_pos: int, dim: int) -> mindspore.Tensor:
    """
    Create sinusoidal positions within a tensor.
    
    Args:
        num_pos (int): The number of positions to create within the tensor.
        dim (int): The dimension of the tensor.
    
    Returns:
        mindspore.Tensor:
            A tensor containing sinusoidal positions. The shape of the tensor is (num_pos, dim).
    
    Raises:
        None.
    """
    inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2, dtype=mindspore.float32) / dim))
    sinusoid_inp = ops.einsum("i , j -> i j", ops.arange(num_pos, dtype=mindspore.int64).float(), inv_freq).float()
    return ops.cat((ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)), axis=1)


# Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
@mindspore.jit
def rotate_every_two(x: mindspore.Tensor) -> mindspore.Tensor:
    """
    Rotate every two elements of the input tensor along the last dimension.
    
    Args:
        x (mindspore.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels,
            H is the height, and W is the width. Must be a 4-dimensional tensor.
    
    Returns:
        mindspore.Tensor:
            A tensor of the same shape as the input tensor after rotating every two elements along the last dimension.
    
    Raises:
        NotImplementedError: If the input tensor is not 4-dimensional.
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = ops.stack((-x2, x1), axis=-1)
    return x.flatten(start_dim=-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


# Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor: mindspore.Tensor, sin: mindspore.Tensor, cos: mindspore.Tensor) -> mindspore.Tensor:
    """
    Apply rotary positional embedding to the input tensor.
    
    Args:
        tensor (mindspore.Tensor): The input tensor to which the positional embedding will be applied.
        sin (mindspore.Tensor): Sine values used for the positional embedding.
        cos (mindspore.Tensor): Cosine values used for the positional embedding.
    
    Returns:
        mindspore.Tensor: A new tensor with the rotary positional embedding applied.
    
    Raises:
        None.
    """
    sin = ops.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = ops.repeat_interleave(cos[:, :, None, :], 2, 3)

    return (tensor * cos) + (rotate_every_two(tensor) * sin)


class CodeGenAttention(nn.Cell):

    ''' 
    Represents an attention mechanism for code generation tasks using the specified configuration.
    
    This class inherits from the nn.Cell module and implements the attention mechanism for code generation tasks.
    It includes methods for splitting and merging attention heads, performing attention calculations, and constructing
    the attention mechanism using the specified inputs.

    The class includes methods for splitting and merging attention heads, performing attention calculations,
    and constructing the attention mechanism using the specified inputs.
    It also handles positional embeddings and caching for efficient computation.

    The methods included in the class are:

    - __init__(self, config): Initializes the CodeGenAttention class with the specified configuration.
    - _split_heads(self, x, n_head, dim_head, mp_num): Splits the attention heads based on the specified parameters.
    - _merge_heads(self, tensor, num_attention_heads, attn_head_size): Merges the attention heads into the specified shape.
    - _attn(self, query, key, value, attention_mask=None, head_mask=None):
        Performs the attention calculation using the query, key, and value tensors, with optional attention and head masks.
    - construct(self, hidden_states, layer_past=None, attention_mask=None, position_ids=None, head_mask=None, use_cache=False, output_attentions=False):
        Constructs the attention mechanism using the specified inputs and optional configurations.

    The CodeGenAttention class provides a comprehensive solution for implementing attention mechanisms in code generation tasks,
    allowing for efficient computation and flexibility in handling various input configurations.
    '''
    def __init__(self, config):
        """
        Initializes an instance of the CodeGenAttention class.

        Args:
            self (CodeGenAttention): The instance of the CodeGenAttention class.
            config: A configuration object that contains various parameters for the attention mechanism.

        Returns:
            None.

        Raises:
            ValueError: If the `embed_dim` is not divisible by `num_attention_heads`.

        """
        super().__init__()

        max_positions = config.max_position_embeddings
        self.causal_mask = ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.bool_)).view(
                1, 1, max_positions, max_positions)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = ops.sqrt(mindspore.tensor(self.head_dim, dtype=mindspore.float32)).to(get_default_dtype())
        self.qkv_proj = nn.Dense(self.embed_dim, self.embed_dim * 3, has_bias=False)

        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    def _split_heads(self, x, n_head, dim_head, mp_num):
        """
        Splits the input tensor into multiple heads for parallel processing in the CodeGenAttention class.

        Args:
            self (CodeGenAttention): An instance of the CodeGenAttention class.
            x (Tensor): The input tensor to be split. It should have a shape of (batch_size, seq_len, hidden_dim).
            n_head (int): The total number of attention heads.
            dim_head (int): The dimension of each attention head.
            mp_num (int): The number of parallel processes.

        Returns:
            reshaped (Tensor): The reshaped tensor after splitting the input tensor into multiple heads.
                It has a shape of (batch_size, seq_len, n_head // mp_num, dim_head).

        Raises:
            None.
        """
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4)
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.shape[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        """
        This method computes the attention mechanism for the CodeGenAttention class.

        Args:
            self: The instance of the CodeGenAttention class.
            query (Tensor): The query tensor with shape [batch_size, query_length, hidden_size].
            key (Tensor): The key tensor with shape [batch_size, key_length, hidden_size].
            value (Tensor): The value tensor with shape [batch_size, key_length, hidden_size].
            attention_mask (Tensor, optional): A mask tensor to mask certain connections during attention computation.
            head_mask (Tensor, optional): A mask tensor to mask certain attention heads.

        Returns:
            tuple (Tensor, Tensor): A tuple containing the attention output tensor with shape
                [batch_size, query_length, hidden_size] and the attention weights tensor with shape
                [batch_size, query_length, key_length].

        Raises:
            ValueError: If the shapes of query, key, or value tensors are incompatible.
            TypeError: If any of the input tensors have incorrect data types.
            RuntimeError: If any unexpected error occurs during the computation.
        """
        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(mindspore.float32)
        key = key.to(mindspore.float32)

        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = finfo(attn_weights.dtype, 'min')
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        mask_value = mindspore.tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = ops.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
        self,
        hidden_states: Optional[mindspore.Tensor],
        layer_past: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[mindspore.Tensor, Tuple[mindspore.Tensor]],
        Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, ...]]],
    ]:
        """
        Constructs the attention mechanism for code generation.

        Args:
            self: The object itself.
            hidden_states (mindspore.Tensor): The hidden states of the input sequence. It is an optional parameter.
            layer_past (Tuple[mindspore.Tensor], optional):
                The past layer's key and value tensors. It is an optional parameter. Defaults to None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. It is an optional parameter.
                Defaults to None.
            position_ids (mindspore.Tensor, optional): The position IDs tensor. It is an optional parameter.
                Defaults to None.
            head_mask (mindspore.Tensor, optional): The head mask tensor. It is an optional parameter. Defaults to None.
            use_cache (bool, optional): Whether to use caching mechanism. It is an optional parameter. Defaults to False.
            output_attentions (bool, optional): Whether to output attention weights. It is an optional parameter.
                Defaults to False.

        Returns:
            Union[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor]], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, ...]]]]:
                The output of the attention mechanism. It can be a tuple containing attention output and present tensors,
                or a tuple containing attention output, present tensors, and attention weights.

        Raises:
            None
        """
        qkv = self.qkv_proj(hidden_states)
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = ops.split(qkv_split, local_dim, axis=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        embed_positions = self.embed_positions
        embed_positions = embed_positions.astype(hidden_states.dtype)

        # sincos = embed_positions[position_ids]
        position_ids_shape = position_ids.shape
        sincos = embed_positions[position_ids.reshape(-1)].reshape(position_ids_shape + (embed_positions.shape[-1],))

        sin, cos = ops.split(sincos, sincos.shape[-1] // 2, axis=-1)

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = ops.cat([k_rot, k_pass], axis=-1)
            query = ops.cat([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            # Note that this cast is quite ugly, but is not implemented before ROPE as k_rot in the original codebase is always in fp32.
            # Reference: https://github.com/salesforce/CodeGen/blob/f210c3bb1216c975ad858cd4132c0fdeabf4bfc2/codegen1/jaxformer/hf/codegen/modeling_codegen.py#L38
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


# Copied from transformers.models.gptj.modeling_gptj.GPTJMLP with GPTJ->CodeGen
class CodeGenMLP(nn.Cell):

    """
    A class that implements a multi-layer perceptron (MLP) for code generation in machine learning tasks.

    This class, named 'CodeGenMLP', is a subclass of the 'nn.Cell' class in the MindSpore framework.
    It is designed to be used in code generation applications within machine learning pipelines.

    The 'CodeGenMLP' class consists of several components, including fully connected layers, activation functions,
    and dropout regularization. It takes an input tensor of hidden states and performs a series of operations to
    transform and refine the data.

    Attributes:
        fc_in (nn.Dense): A fully connected layer that maps the input tensor to an intermediate size.
        fc_out (nn.Dense): A fully connected layer that maps the intermediate tensor to the output size.
        act (ACT2FN): An activation function defined in the 'config.activation_function'.
        dropout (nn.Dropout): A dropout layer with a dropout probability defined in 'config.resid_pdrop'.

    Methods:
        __init__(self, intermediate_size, config): Initializes the 'CodeGenMLP' object.

            - intermediate_size (int): The size of the intermediate layer in the MLP.
            - config (object): A configuration object containing various settings for the MLP.

        construct(self, hidden_states: Optional[mindspore.Tensor]) -> mindspore.Tensor: Constructs the forward pass of the MLP.

            - hidden_states (mindspore.Tensor): An optional input tensor of hidden states.
            - Returns:
                - mindspore.Tensor: The output tensor after passing through the MLP.

    Example:
        ```python
        >>> # Create an instance of the 'CodeGenMLP' class
        >>> mlp = CodeGenMLP(intermediate_size, config)
        ...
        >>> # Pass an input tensor through the MLP
        >>> output = mlp.construct(hidden_states)
        ```
    """
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        """
        Initializes an instance of the CodeGenMLP class.

        Args:
            self (object): The instance of the class.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object containing parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the intermediate_size is not a positive integer.
            KeyError: If the activation function specified in the config is not supported.
            AttributeError: If the configuration object does not contain the required attributes.
            RuntimeError: If there is an issue with initializing the dense layers or dropout.
        """
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Dense(embed_dim, intermediate_size)
        self.fc_out = nn.Dense(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: Optional[mindspore.Tensor]) -> mindspore.Tensor:
        """
        Constructs the forward pass of the CodeGenMLP model.

        Args:
            self (CodeGenMLP): An instance of the CodeGenMLP class.
            hidden_states (Optional[mindspore.Tensor]): The input hidden states tensor.
                It represents the features of the input data.
                The shape of the tensor should be compatible with the input layer of the model.

        Returns:
            mindspore.Tensor: The output tensor after applying the forward pass operations.
                The shape of the tensor will depend on the architecture of the model.

        Raises:
            None.

        Note:
            This method performs the following steps:

            1. Applies a fully connected layer to the input hidden states.
            2. Applies an activation function to the output of the previous layer.
            3. Applies another fully connected layer to the output of the activation function.
            4. Applies dropout regularization to the output of the previous layer.
            5. Returns the final output tensor after the forward pass operations.
        """
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.gptj.modeling_gptj.GPTJBlock with GPTJ->CodeGen
class CodeGenBlock(nn.Cell):

    """
    This class represents a code generation block in a neural network model.
    It is a subclass of nn.Cell and contains methods for initializing and constructing the block.

    Attributes:
        ln_1 (nn.LayerNorm): The layer normalization module applied to the input hidden states.
        attn (CodeGenAttention): The attention module used for generating attention outputs.
        mlp (CodeGenMLP): The multi-layer perceptron module used for feed-forward computation.

    Methods:
        __init__(self, config):
            Initializes a CodeGenBlock instance.

            Args:

            - config: A configuration object containing the necessary parameters for the block initialization.

        construct(self, hidden_states, layer_past=None, attention_mask=None, position_ids=None, head_mask=None, use_cache=False, output_attentions=False):
            Constructs the block by performing attention mechanism, feed-forward computation, and residual addition.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states to the block.
            - layer_past (Tuple[mindspore.Tensor], optional): The previous layer's hidden states. Defaults to None.
            - attention_mask (mindspore.Tensor, optional): The attention mask tensor. Defaults to None.
            - position_ids (mindspore.Tensor, optional): The position IDs tensor. Defaults to None.
            - head_mask (mindspore.Tensor, optional): The head mask tensor. Defaults to None.
            - use_cache (bool, optional): Whether to use cache for storing attention outputs. Defaults to False.
            - output_attentions (bool, optional): Whether to output attention matrices. Defaults to False.

            Returns:

            - Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
            - The constructed block outputs, which include the final hidden states and optionally attention matrices.
    """
    def __init__(self, config):
        """
        Initializes an instance of the CodeGenBlock class.

        Args:
            self: The instance of the CodeGenBlock class.
            config:
                A configuration object containing parameters for the block.

                - Type: object
                - Purpose: The configuration object to customize the block's behavior.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon)
        self.attn = CodeGenAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)

    def construct(
        self,
        hidden_states: Optional[mindspore.Tensor],
        layer_past: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
        """
        Constructs a CodeGenBlock by processing the input hidden states through attention mechanism and feed forward network.

        Args:
            self (CodeGenBlock): The instance of the CodeGenBlock class.
            hidden_states (Optional[mindspore.Tensor]): The input hidden states to be processed.
            layer_past (Optional[Tuple[mindspore.Tensor]]): Optional past layer hidden states for recurrence.
            attention_mask (Optional[mindspore.Tensor]): Optional mask to prevent attention to certain positions.
            position_ids (Optional[mindspore.Tensor]): Optional tensor specifying the position ids.
            head_mask (Optional[mindspore.Tensor]): Optional mask to prevent attention to certain heads.
            use_cache (Optional[bool]): Flag indicating whether to use cache for faster decoding.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.

        Returns:
            Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
                Depending on the use_cache flag, returns the processed hidden states and optionally the intermediate outputs.

        Raises:
            None
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class CodeGenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CodeGenConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["CodeGenBlock"]

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


class CodeGenModel(CodeGenPreTrainedModel):

    """
    The `CodeGenModel` class is a Python class that represents a code generation model.
    It is a subclass of `CodeGenPreTrainedModel` and provides functionality for constructing, setting and getting input
    embeddings, and generating code outputs.

    Attributes:
        `embed_dim`: An integer representing the embedding dimension.
        `vocab_size`: An integer representing the size of the vocabulary.
        `wte`: An embedding layer that maps input tokens to their corresponding embeddings.
        `drop`: A dropout layer for regularization.
        `h`: A list of `CodeGenBlock` instances representing code generation blocks.
        `ln_f`: A layer normalization layer.
        `rotary_dim`: An integer representing the dimension of the rotary encoding.
        `gradient_checkpointing`: A boolean indicating whether gradient checkpointing is enabled.

    Note:
        Please refer to the `CodeGenPreTrainedModel` class for additional inherited attributes and methods.

    """
    def __init__(self, config):
        """Initializes an instance of the CodeGenModel class.

        Args:
            self: The instance of the class.
            config:
                An object of the configuration class containing the following attributes:

                - n_embd (int): The embedding dimension.
                - vocab_size (int): The size of the vocabulary.
                - embd_pdrop (float): The dropout probability for the embeddings.
                - n_layer (int): The number of code generation blocks.
                - layer_norm_epsilon (float): The epsilon value for layer normalization.
                - rotary_dim (int): The dimension of the rotary positional embeddings.
                - n_ctx (int): The context length.
                - num_attention_heads (int): The number of attention heads.
            The config object is used to initialize various attributes of the CodeGenModel instance.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([CodeGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method is part of the CodeGenModel class and is named get_input_embeddings.
        It takes 1 parameter, self, which refers to the instance of the class.

        Args:
            self (CodeGenModel): The instance of the CodeGenModel class.

        Returns:
            None.

        Raises:
            This method does not raise any exceptions.
        """
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the CodeGenModel.

        Args:
            self (CodeGenModel): The instance of the CodeGenModel class.
            new_embeddings: The new embeddings to be set. This argument could be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.wte = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the CodeGenModel.

        Args:
            self (CodeGenModel): The instance of the CodeGenModel class.
            input_ids (Optional[mindspore.Tensor]): The input IDs of the model. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values for the model. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask for the model. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The token type IDs for the model. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs for the model. Default is None.
            head_mask (Optional[mindspore.Tensor]): The head mask for the model. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs for the model. Default is None.
            use_cache (Optional[bool]): Whether to use cache. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return the result as a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The constructed model output.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If batch_size is less than or equal to zero.
            Warning: If use_cache is True and config.gradient_checkpointing is True.
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
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = ops.arange(past_length, input_shape[-1] + past_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)

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
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * finfo(self.dtype, 'min')

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
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


class CodeGenForCausalLM(CodeGenPreTrainedModel):

    """
    This class represents a code generation model for causal language modeling (LM) using a transformer-based architecture.
    It inherits from the CodeGenPreTrainedModel class.

    Attributes:
        transformer (CodeGenModel): The transformer model for code generation.
        lm_head (nn.Dense): The dense layer for predicting the next token in the language modeling task.

    Methods:
        __init__: Initializes the CodeGenForCausalLM instance.
        get_output_embeddings: Returns the output embeddings of the LM head.
        set_output_embeddings: Sets the output embeddings of the LM head.
        prepare_inputs_for_generation: Prepares the input tensors for generation.
        construct: Constructs the output of the model for a given set of inputs.
        _reorder_cache: Reorders the cache of past key values for beam search or beam sampling.

    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the CodeGenForCausalLM class.

        Args:
            self: The instance of the CodeGenForCausalLM class.
            config: An object containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            This method does not explicitly raise any exceptions.
        """
        super().__init__(config)
        self.transformer = CodeGenModel(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method 'get_output_embeddings' in the class 'CodeGenForCausalLM' retrieves the output embeddings from the model's head.

        Args:
            self: The instance of the class 'CodeGenForCausalLM' itself.

        Returns:
            lm_head: This method returns the output embeddings represented by 'self.lm_head', which is of type 'None'.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the CodeGenForCausalLM.

        Args:
            self (CodeGenForCausalLM): The instance of the CodeGenForCausalLM class.
            new_embeddings: The new embeddings to be set as the output embeddings.
                It should be of the same type as the current embeddings.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method replaces the current output embeddings of the CodeGenForCausalLM instance
            with the provided new embeddings. The new embeddings should have the same type as the
            current embeddings to ensure compatibility with the other methods of the class.
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare inputs for generation.

        Args:
            self (CodeGenForCausalLM): The instance of the CodeGenForCausalLM class.
            input_ids (tensor): The input tensor containing token IDs for the model.
            past_key_values (tuple or None): Tuple of past key values used for fast decoding.
                Defaults to None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids': The trimmed input tensor ready for generation.
                - 'past_key_values': The past_key_values parameter passed to the method.
                - 'use_cache': Boolean flag indicating whether to use caching for faster decoding.
                - 'position_ids': Tensor containing positional IDs for the input tokens.
                - 'attention_mask': Tensor containing attention mask for the input.
                - 'token_type_ids': Tensor containing token type IDs if available.

        Raises:
            None.
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(mindspore.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

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
    def _reorder_cache(
        past_key_values: Tuple[Tuple[mindspore.Tensor]], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )

__all__ = [
    "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
    "CodeGenForCausalLM",
    "CodeGenModel",
    "CodeGenPreTrainedModel",
]
