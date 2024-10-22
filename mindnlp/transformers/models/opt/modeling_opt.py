# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
# pylint: disable=unexpected-keyword-arg
""" MindSpore OPT model."""
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import Normal, initializer

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from .configuration_opt import OPTConfig
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://hf-mirror.com/models?filter=opt
]


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes an instance of the OPTLearnedPositionalEmbedding class with the given parameters.
        
        Args:
            self (OPTLearnedPositionalEmbedding): The instance of the class.
            num_embeddings (int): The number of embeddings.
                This parameter specifies the size of the embedding table.
                It must be a positive integer.
            embedding_dim (int): The dimensionality of the embeddings.
                This parameter determines the size of each embedding vector.
                It must be a positive integer.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: mindspore.Tensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (ops.cumsum(attention_mask, dim=1).astype(attention_mask.dtype) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        """
        Initialize an OPTAttention object.
        
        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability, defaults to 0.0.
            is_decoder (bool, optional): Indicates if the attention is used in a decoder, defaults to False.
            bias (bool): Indicates whether bias is added in linear transformations.
        
        Returns:
            None.
        
        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor according to the specified dimensions.
        
        Args:
            self (OPTAttention): An instance of the OPTAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.
            
        Returns:
            None: This method modifies the input tensor in-place.
            
        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        key_value_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = ops.maximum(
                attn_weights, mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == mindspore.float16:
            attn_weights = ops.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(mindspore.float16)
        else:
            attn_weights = ops.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OPTDecoderLayer(nn.Module):

    """
    OPTDecoderLayer is a class that represents a single layer of the OPT (Optimized Performance Transformer) decoder model. 
    It implements the decoding logic for the transformer model and includes self-attention mechanism,
    feedforward neural network, and layer normalization.
    
    This class inherits from nn.Module and is designed to be used within a transformer decoder stack for
    sequence-to-sequence tasks.
    
    Attributes:
        embed_dim (int): The dimension of the hidden states in the layer.
        self_attn (OPTAttention): The self-attention mechanism used in the layer.
        do_layer_norm_before (bool): Flag indicating whether layer normalization is applied before the
            self-attention operation.
        dropout (float): The dropout rate applied to the layer.
        activation_fn (function): The activation function used in the feedforward neural network.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization applied to the self-attention output.
        fc1 (nn.Linear): The first linear transformation in the feedforward neural network.
        fc2 (nn.Linear): The second linear transformation in the feedforward neural network.
        final_layer_norm (nn.LayerNorm): Layer normalization applied to the final output of the layer.

    Methods:
        forward(hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions, use_cache):
          Constructs the output of the decoder layer given the input hidden states and optional arguments.

    Args:
        hidden_states (mindspore.Tensor): The input to the layer of shape `(batch, seq_len, embed_dim)`.
        attention_mask (mindspore.Tensor, optional): The attention mask of size `(batch, 1, tgt_len, src_len)`.
        layer_head_mask (mindspore.Tensor, optional): The mask for attention heads in a given layer.
        past_key_value (Tuple(mindspore.Tensor), optional): Cached past key and value projection states.
        output_attentions (bool, optional): Whether to return the attention tensors of all attention layers.
        use_cache (bool, optional): If set to True, past key value states are returned for speeding up decoding.

    Returns:
        outputs (Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]):
            The output tensor of the decoder layer.

            - If output_attentions is True, self-attention weights are also included in the output.
            - If use_cache is True, present key value states are also included in the output.
    """
    def __init__(self, config: OPTConfig):
        """Initializes an instance of the OPTDecoderLayer class.

        Args:
            self (OPTDecoderLayer): The instance of the class.
            config (OPTConfig):
                The configuration object containing the parameters for the layer.

                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - attention_dropout (float): The dropout rate for attention layers.
                - is_decoder (bool): Specifies if the layer is used as a decoder.
                - enable_bias (bool): Specifies if bias is enabled.
                - do_layer_norm_before (bool): Specifies if layer normalization is applied before the
                self-attention layer.
                - dropout (float): The dropout rate for the layer.
                - activation_function (str): The activation function to be used.
                - layer_norm_elementwise_affine (bool): Specifies if elementwise affine transformation is used in
                layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            [self.embed_dim], elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim], elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OPTPreTrainedModel(PreTrainedModel):

    """
    This class represents a pre-trained model for optimization tasks in natural language processing (NLP).
    It is a subclass of the PreTrainedModel class.

    The OPTPreTrainedModel class provides methods and attributes for initializing the weights of different types of
    cells, such as Dense and Embedding. The weights are initialized using a specified standard deviation, and biases
    are set to zero if present.

    Methods:
        _init_weights: Initializes the weights of the given cell based on the specified standard deviation.
            For Dense cells, the weight data is set using the initializer function with a Normal distribution.
            If the cell has biases, they are initialized to zero. For Embedding cells, the weight data is
            filled with random values from a Normal distribution, and if a padding index is provided, the
            corresponding weight is set to zero.

    Attributes:
        config: An instance of a configuration class that stores various settings and hyperparameters for
            the pre-trained model.

    Note:
        This class is designed to be used as a base class for specific optimization tasks in NLP.
        It does not implement any specific optimization algorithms or provide training or inference functionality.
    """
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]

    def _init_weights(self, cell):
        """
        Initializes the weights of a neural network cell.

        Args:
            self: An instance of the OPTPreTrainedModel class.
            cell: The neural network cell whose weights are to be initialized.
                This can be an instance of nn.Linear or nn.Embedding.

        Returns:
            None.

        Raises:
            None.

        This method initializes the weights of the specified neural network cell. If the cell is an instance of nn.Linear,
        the weight is initialized using the Normal initializer with standard deviation 'self.config.init_std'.
        If the cell has a bias, the bias is initialized with zeros.
        If the cell is an instance of nn.Embedding, the weight is initialized with random values drawn from a
        normal distribution with mean 0 and standard deviation 'self.config.init_std'.
        If the cell has a padding index, the weight at the padding index is set to 0.
        """
        std = self.config.init_std
        if isinstance(cell, nn.Linear):
            cell.weight.assign_value(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.assign_value(Tensor(weight, cell.weight.dtype))


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """
    def __init__(self, config: OPTConfig):
        """
        Initializes an instance of the OPTDecoder class.

        Args:
            self: The instance of the class.
            config: An instance of the OPTConfig class containing configuration parameters for the decoder.

        Returns:
            None

        Raises:
            None

        The method initializes the decoder with the given configuration parameters.
        It sets the dropout and layerdrop values, the padding index, the maximum target positions,
        and the vocabulary size. It also initializes the embedding tokens and positions with the given configuration
        parameters. If the word embedding projection dimension is not equal to the hidden size,
        it initializes the projection layers. It also initializes the final layer normalization if the configuration
        parameters allow it.
        Finally, it creates a set of OPTDecoderLayer instances, one for each hidden layer in the configuration.
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, padding_idx=self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                [config.hidden_size], elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings.

        Args:
            self: An instance of the OPTDecoder class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the OPTDecoder instance.
        The input embeddings are used for further processing in the decoding algorithm.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method: set_input_embeddings

        Description:
            Sets the input embeddings for the OPTDecoder instance.

        Args:
            self (OPTDecoder): The instance of OPTDecoder.
            value (Any): The input embeddings to be set. It can be of any type.

        Returns:
            None.

        Raises:
            TypeError: If the input embeddings value is not of the expected type.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(batch_size, mask_seq_length)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class OPTModel(OPTPreTrainedModel):

    """
    The `OPTModel` class represents an OPT (Orphaned Pretrained Transformer) model, which is a specific type of 
    pre-trained transformer model used for various natural language processing tasks. This class inherits from the 
    `OPTPreTrainedModel` class.

    Attributes:
        `decoder`: An instance of `OPTDecoder` class representing the decoder component of the OPT model.

    Methods:
        `__init__`: Constructs a new OPTModel instance by initializing the superclass and setting up the decoder component.
        `get_input_embeddings`: Retrieves the input embeddings used by the decoder.
        `set_input_embeddings`: Sets the input embeddings of the decoder.
        `get_decoder`: Retrieves the decoder instance.
        `forward`: Constructs the OPT model by calling the decoder with the provided input parameters. 
            It returns the decoder outputs, which can include the last hidden state, past key values, hidden states, 
            and attentions.

    Note:
        - The forwardor `__init__` should be called to initialize a new instance of `OPTModel` with a `config` object.
        - The `forward` method is the main method to generate the outputs of the OPT model based on the given inputs.
        - The other methods are used to retrieve or modify specific components of the model.

    """
    def __init__(self, config: OPTConfig):
        """
        Initializes an OPTModel object with the provided configuration.

        Args:
            self (OPTModel): The instance of the OPTModel class.
            config (OPTConfig): An instance of OPTConfig containing configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the OPTModel's decoder.

        Args:
            self (OPTModel): The instance of OPTModel class itself.
                This parameter is required to access the decoder's embed_tokens.

        Returns:
            embed_tokens: The method returns the embeddings from the decoder's embed_tokens attribute.

        Raises:
            None.
        """
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the OPTModel.

        Args:
            self (OPTModel): The instance of the OPTModel class.
            value: The input embeddings to be set.
                It should be an object of type `torch.nn.Embedding` or a subclass of it.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the OPTModel's decoder.
        The decoder's `embed_tokens` attribute is updated with the provided `value`.
        The `value` should be an instance of `torch.nn.Embedding` or a subclass of it.
        This allows for customizing the input embeddings used by the decoder during the model's forward pass.
        """
        self.decoder.embed_tokens = value

    def get_decoder(self):
        """
        Returns the decoder of the OPTModel.

        Args:
            self (OPTModel): An instance of the OPTModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the model for OPTModel.

        Args:
            self (OPTModel): The instance of the OPTModel class.
            input_ids (mindspore.Tensor): The input tensor containing the token IDs. Default is None.
            attention_mask (Optional[mindspore.Tensor]): Tensor representing the attention mask. Default is None.
            head_mask (Optional[mindspore.Tensor]): Tensor representing the head mask. Default is None.
            past_key_values (Optional[List[mindspore.Tensor]]): List of tensors containing past key values.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): Tensor containing embedded inputs. Default is None.
            use_cache (Optional[bool]): Flag indicating whether to use caching. Default is None.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output of the model forwardion.

        Raises:
            None
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    r"""
    This class represents an OPT (Optimus) model for Causal Language Modeling (LM), which is used for generating text
    based on given input sequences.
    The class includes methods for initializing the model, getting and setting input and output embeddings,
    setting and getting the decoder, forwarding the model, and preparing inputs for text generation.
    It inherits from OPTPreTrainedModel and provides functionalities for handling various parameters related to
    text generation tasks.

    Methods:
        __init__: Initialize the OPTForCausalLM model with a given configuration.
        get_input_embeddings: Get the input embeddings from the model's decoder.
        set_input_embeddings: Set new input embeddings for the model's decoder.
        get_output_embeddings: Get the output embeddings from the model's lm_head layer.
        set_output_embeddings: Set new output embeddings for the model's lm_head layer.
        set_decoder: Set a new decoder for the model.
        get_decoder: Get the current decoder used in the model.
        forward: Construct the model for text generation with various input parameters and return the output.
        prepare_inputs_for_generation:
            Prepare inputs for text generation by handling past key values and attention masks.
        _reorder_cache(past_key_values, beam_idx):
            Reorder the cache elements based on the beam index for efficient decoding.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM
        ...
        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        ...
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="ms")
        ...
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of OPTForCausalLM.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is not of the expected type.
            ValueError: If the configuration parameters are invalid or missing.
            RuntimeError: If an error occurs during model initialization or post-initialization steps.
        """
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the OPTForCausalLM model.

        Args:
            self: An instance of the OPTForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the OPTForCausalLM model.

        Args:
            self (OPTForCausalLM): The instance of the OPTForCausalLM class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a torch.Tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the OPTForCausalLM model.

        Args:
            self: An instance of the OPTForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves and returns the output embeddings of the OPTForCausalLM model.
        The output embeddings are the final layer of the model, which are responsible for generating the predictions or
        outputs based on the input sequences. The output embeddings capture the learned representation of the input
        sequence and can be used for various downstream tasks.

        Note: The output embeddings are accessed through the 'lm_head' attribute of the OPTForCausalLM instance.

        Example:
            ```python
            >>> model = OPTForCausalLM()
            >>> output_embeddings = model.get_output_embeddings()
            ```
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the OPTForCausalLM model.

        Args:
            self (OPTForCausalLM): The instance of the OPTForCausalLM class.
            new_embeddings (tensor): The new output embeddings to be set for the model.
                It should be a tensor of the appropriate shape and type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Method to set the decoder for the OPTForCausalLM class.

        Args:
            self (OPTForCausalLM): The instance of the OPTForCausalLM class.
                This parameter represents the current instance of the class.
            decoder (object): The decoder object to be set for the model.
                This parameter is the decoder object that will be assigned to the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Returns the decoder model used for the OPTForCausalLM instance.

        Args:
            self: The OPTForCausalLM instance itself.

        Returns:
            None

        Raises:
            None
        """
        return self.model.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, OPTForCausalLM
            ...
            >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            ...
            >>> prompt = "Hey, are you conscious? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="ms")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (OPTForCausalLM): The instance of the OPTForCausalLM class.
            input_ids (torch.Tensor): The input tensor of token IDs representing the input sequence.
            past_key_values (tuple, optional): The tuple of past key values used for generation. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor indicating which tokens should be
                attended to. Defaults to None.
            inputs_embeds (torch.Tensor, optional): The input tensor of token embeddings representing the input
                sequence. Defaults to None.

        Returns:
            dict:
                A dictionary containing the model inputs for generation. The dictionary can have the following keys:

                - 'inputs_embeds': The input tensor of token embeddings if inputs_embeds is not None
                and past_key_values is None.
                - 'input_ids': The input tensor of token IDs if inputs_embeds is None or past_key_values is not None.
                - 'past_key_values': The tuple of past key values used for generation.
                - 'use_cache': The flag indicating whether to use cache during generation.
                - 'attention_mask': The attention mask tensor indicating which tokens should be attended to.

        Raises:
            None.
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the OPTForCausalLM class based on the specified beam index.

        Args:
            past_key_values (tuple): A tuple of past key-value states for each layer.
                Each layer's past state is a tensor of shape (batch_size, sequence_length, hidden_size).
            beam_idx (Tensor): A tensor representing the indices of beams to use for reordering.
                It has a shape of (batch_size, num_beams).

        Returns:
            reordered_past (tuple): A tuple of reordered past key-value states for each layer.
                Each layer's reordered past state is a tensor of shape
                (batch_size * num_beams, sequence_length, hidden_size).

        Raises:
            None.

        Note:
            The method assumes that the past_key_values and beam_idx are compatible and have appropriate dimensions.

        Example:
            >>> past_key_values = (layer1_past, layer2_past, ...)
            >>> beam_idx = tensor([[0, 2, 1], [1, 0, 2]])
            >>> reordered_past = _reorder_cache(past_key_values, beam_idx)
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class OPTForSequenceClassification(OPTPreTrainedModel):

    """
    OPTForSequenceClassification

    This class is a sequence classification model based on the OPT (OpenAI's Pretrained Transformer) architecture.
    It inherits from OPTPreTrainedModel and provides functionalities for sequence classification tasks.

    Attributes:
        num_labels (int): The number of labels for the classification task.
        model (OPTModel): The OPTModel instance that serves as the core transformer model.
        score (nn.Linear): A fully connected layer that maps the transformer outputs to the number of labels.

    Methods:
        __init__: Initializes the OPTForSequenceClassification instance.
        forward: Constructs the sequence classification model and returns the output.
        get_input_embeddings: Returns the embedding layer for the input tokens.
        set_input_embeddings: Sets the embedding layer for the input tokens.

    """
    def __init__(self, config: OPTConfig):
        """
        Initializes an instance of the OPTForSequenceClassification class.

        Args:
            self: The instance of the class.
            config (OPTConfig):
                The configuration object containing parameters for initializing the model.

                - num_labels (int): The number of labels for the classification task.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
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
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

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
                if self.num_labels == 1:
                    loss = F.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(pooled_logits, labels)
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

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the OPTForSequenceClassification model.

        Args:
            self: An instance of the OPTForSequenceClassification class.

        Returns:
            embed_tokens: The method returns the input embeddings for the model's decoder.
                The input embeddings are used to represent the input tokens in the model.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the OPTForSequenceClassification model.

        Args:
            self (OPTForSequenceClassification): The instance of the OPTForSequenceClassification class.
            value: The input embeddings to be set for the model. This should be a tensor representing the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value


class OPTForQuestionAnswering(OPTPreTrainedModel):

    '''
    This class represents a question answering model using the OPT (OpenAI's Public Tenders) architecture.
    It inherits from OPTPreTrainedModel and provides methods for model forwardion, obtaining input
    embeddings, and setting input embeddings. The model is designed to take in various inputs, such as input IDs,
    attention masks, head masks, past key values, and inputs embeddings, and return outputs for question answering tasks.
    The forward method allows for flexible input options and returns a tuple or a QuestionAnsweringModelOutput
    based on the input and return options. The class also provides methods for accessing and updating the input
    embeddings for the model.
    '''
    def __init__(self, config: OPTConfig):
        """
        Initializes an instance of the OPTForQuestionAnswering class with the provided configuration.

        Args:
            self (OPTForQuestionAnswering): The instance of the OPTForQuestionAnswering class.
            config (OPTConfig): An instance of OPTConfig containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type OPTConfig.
            ValueError: If any of the configuration settings provided in config are invalid or missing.
            RuntimeError: If there is an issue during the initialization process.
        """
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
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

        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
            ...
            >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            ...
            >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
            >>> # so the head will be randomly initialized, hence the predictions will be random
            >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")
            ...
            >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            ...
            >>> inputs = tokenizer(question, text, return_tensors="ms")
            >>> with torch.no_grad():
            ...     outputs = model(**inputs)
            ...
            >>> answer_start_index = outputs.start_logits.argmax()
            >>> answer_end_index = outputs.end_logits.argmax()
            ...
            >>> answer_offset = len(tokenizer(question)[0])
            ...
            >>> predict_answer_tokens = inputs.input_ids[
            ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
            ... ]
            >>> predicted = tokenizer.decode(predict_answer_tokens)
            >>> predicted
            ' a nice puppet'
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
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

        logits = self.qa_outputs(hidden_states)
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
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the OPTForQuestionAnswering model.
        
        Args:
            self: An instance of the OPTForQuestionAnswering class.
        
        Returns:
            embed_tokens: This method returns the input embeddings obtained from the model's decoder embed_tokens.
        
        Raises:
            None
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the OPTForQuestionAnswering class.
        
        Args:
            self (OPTForQuestionAnswering): The instance of the OPTForQuestionAnswering class.
            value (object): The input embeddings to be set for the model decoder.
                Should be compatible with the decoder's embed_tokens attribute.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

__all__ = [
    "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "OPTForCausalLM",
    "OPTModel",
    "OPTPreTrainedModel",
    "OPTForSequenceClassification",
    "OPTForQuestionAnswering",
]
