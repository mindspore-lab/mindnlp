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
"""MindNLP gpt2 model"""

from typing import Optional, Tuple

import math
import mindspore
import numpy as np
from mindspore import nn, ops, Tensor, Parameter, dtype_to_nptype
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp._legacy.functional import split, where, arange, softmax
from mindnlp._legacy.nn import Dropout, Matmul
from .configuration_gpt2 import GPT2Config
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...ms_utils import Conv1D, prune_conv1d_layer, find_pruneable_heads_and_indices

logger = logging.get_logger(__name__)

GPT2_SUPPORT_LIST = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"]


__all__ = ['GPT2DoubleHeadsModel', 'GPT2ForSequenceClassification',
           'GPT2ForTokenClassification', 'GPT2LMHeadModel', 'GPT2Model']


class GPT2Attention(nn.Cell):
    r"""
    gpt2 Attention
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        """
        Initialize the GPT2Attention class.
        
        Args:
            self: The object instance.
            config: An object containing configuration settings for the GPT2Attention model.
            is_cross_attention (bool): Flag indicating if the attention mechanism is for cross-attention.
            layer_idx (int or None): Index of the layer.
        
        Returns:
            None.
        
        Raises:
            ValueError: Raised if the `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()

        max_positions = config.max_position_embeddings
        self.bias = Parameter(Tensor(np.tril(np.ones((max_positions, max_positions))).reshape(
                (1, 1, max_positions, max_positions)
            ), mindspore.bool_), requires_grad=False)
        self.masked_bias = Tensor(-1e4, mindspore.float32)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = Dropout(p=config.attn_pdrop)
        self.resid_dropout = Dropout(p=config.resid_pdrop)

        self.pruned_heads = set()
        self.output_attentions = config.output_attentions
        self.matmul = Matmul()
        self.mask_value = Tensor(np.finfo(np.float32).min, dtype=mindspore.float32)

    def prune_heads(self, heads):
        """
        Prunes heads of the model.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, axis=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, axis=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Method _attn in the class GPT2Attention.
        
        Args:
            self: GPT2Attention object
                The instance of the GPT2Attention class.
            query: Tensor
                The input query tensor.
            key: Tensor
                The input key tensor.
            value: Tensor
                The input value tensor.
            attention_mask: Tensor or None
                Optional tensor for masking attention scores.
            head_mask: Tensor or None
                Optional tensor for applying head-level mask.

        Returns:
            Tuple (Tensor, Tensor):
                Tuple containing the attention output tensor and the attention weights tensor.

        Raises:
            ValueError:
                If the input tensors have incompatible shapes.
            TypeError:
                If any of the input tensors have incorrect data types.
            RuntimeError:
                If an error occurs during the attention calculation process.
        """
        attn_weights = self.matmul(query, key.swapaxes(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / ops.sqrt(ops.scalar_to_tensor(value.shape[-1]))

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            multiplu_out = Tensor(1.0, mindspore.float32) - causal_mask
            adder = multiplu_out * self.masked_bias
            attn_weights = ops.add(attn_weights, adder)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax(attn_weights, axis=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = self.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        This method '_upcast_and_reordered_attn' is a part of the 'GPT2Attention' class and performs upcasting and
        reordering operations on the provided query, key, and value tensors to compute the attention weights and output.
        The method takes the following parameters:

        Args:
            self: The instance of the class.
            query: A tensor representing the query input with shape
                (batch_size, num_heads, query_sequence_length, hidden_size).
            key: A tensor representing the key input with shape
                (batch_size, num_heads, key_sequence_length, hidden_size).
            value: A tensor representing the value input with shape
                (batch_size, num_heads, key_sequence_length, hidden_size).
            attention_mask: An optional tensor with the same shape as attn_weights, used to mask the attention scores.
            head_mask: An optional tensor with shape (num_heads,) or (num_layers, num_heads) used to mask the
                attention scores in the multi-head attention mechanism.

        Returns:
            Tuple[Tensor]:
                This method returns two output tensors:

                - attn_output: A tensor representing the output of the attention mechanism with shape
                (batch_size, num_heads, query_sequence_length, hidden_size).
                - attn_weights: A tensor representing the attention weights computed during the attention mechanism
                with shape (batch_size * num_heads, query_sequence_length, key_sequence_length).

        Raises:
            RuntimeError: If the upcasting operation fails, and the attn_weights tensor does not have the required
                dtype mindspore.float32.
        """
        # Use `mindspore.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, _ = query.shape
        _, _, k_seq_len, _ = key.shape

        # Preallocate attn_weights for `baddbmm`
        attn_weights = ops.zeros((bsz * num_heads, q_seq_len, k_seq_len), dtype=mindspore.float32)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.shape[-1]) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.shape[-2], key.shape[-2]
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            mask_value = Tensor(np.finfo(dtype_to_nptype(attn_weights.dtype)).min, dtype=attn_weights.dtype)
            attn_weights = where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax(attn_weights, axis=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != mindspore.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype mindspore.float32")
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = self.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return ops.transpose(tensor, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = ops.transpose(tensor, (0, 2, 1, 3))
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def construct(
            self,
            hidden_states: Tuple[Tensor],
            layer_past: Optional[Tuple[Tensor]] = None,
            attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            use_cache: Optional[bool] = False,
    ):
        """
        This method 'construct' is a part of the 'GPT2Attention' class and is responsible for constructing the
        attention mechanism with various parameters.

        Args:
            self: The instance of the class.
            hidden_states (Tuple[Tensor]): Tuple of tensors representing the hidden states.
            layer_past (Optional[Tuple[Tensor]]): Optional tuple of tensors representing past layer states.
            attention_mask (Optional[Tensor]): Optional tensor representing attention mask.
            head_mask (Optional[Tensor]): Optional tensor representing head mask.
            encoder_hidden_states (Optional[Tensor]): Optional tensor representing encoder hidden states.
            encoder_attention_mask (Optional[Tensor]): Optional tensor representing encoder attention mask.
            use_cache (Optional[bool]): Optional boolean indicating whether to use cache or not.

        Returns:
            None.

        Raises:
            ValueError: Raised if 'encoder_hidden_states' is not None and 'q_attn' weights are not defined.
        """
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = split(self.c_attn(encoder_hidden_states), self.split_size, axis=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = split(self.c_attn(hidden_states), self.split_size, axis=2)

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

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if self.output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Cell):
    r"""
    gpt2 MLP
    """
    def __init__(self, intermediate_size, config):
        """
        Initializes an instance of the GPT2MLP class.

        Args:
            self: The object itself.
            intermediate_size (int): The size of the intermediate layer.
            config (object): The configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = Dropout(p=config.resid_pdrop)

    def construct(self, hidden_states: Tuple[Tensor]):
        """
        Constructs the hidden states in the GPT2MLP class.

        Args:
            self: An instance of the GPT2MLP class.
            hidden_states (Tuple[Tensor]): A tuple of tensors representing the hidden states.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Cell):
    r"""
    gpt2 Block
    """
    def __init__(self, config, layer_idx=None):
        """
        Initializes a GPT2Block object.

        Args:
            self: The instance of the GPT2Block class.
            config:
                An object containing configuration settings for the GPT2Block.

                - Type: Any
                - Purpose: Specifies the configuration settings for the GPT2Block.

            layer_idx:
                An integer representing the index of the layer.

                - Type: int or None
                - Purpose: Specifies the index of the layer. If None, the default value is used.
                - Restrictions: Must be an integer or None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm((hidden_size,), epsilon=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm((hidden_size,), epsilon=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def construct(
            self,
            hidden_states: Tuple[Tensor],
            layer_past: Optional[Tuple[Tensor]] = None,
            attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            use_cache: Optional[bool] = False,
    ):
        """
        Constructs a GPT2Block.

        Args:
            self: The GPT2Block instance.
            hidden_states (Tuple[Tensor]): The hidden states.
            layer_past (Optional[Tuple[Tensor]]): The past hidden states of the layer. Default is None.
            attention_mask (Optional[Tensor]): The attention mask. Default is None.
            head_mask (Optional[Tensor]): The head mask. Default is None.
            encoder_hidden_states (Optional[Tensor]): The hidden states of the encoder. Default is None.
            encoder_attention_mask (Optional[Tensor]): The attention mask of the encoder. Default is None.
            use_cache (Optional[bool]): Whether to use cache or not. Default is False.

        Returns:
            Tuple[Tensor]: The outputs of the GPT2Block.

        Raises:
            ValueError: If `encoder_hidden_states` are passed, and the GPT2Block is not instantiated with cross-attention
                layers by setting `config.add_cross_attention=True`.

        Note:
            The method updates the hidden states and applies attention mechanisms and feed-forward networks to the
            input states. It also handles cross-attention if `encoder_hidden_states` are provided.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
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
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

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


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config

    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """
        Prepare the head mask if needed.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.expand_dims(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.astype(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

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

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in cell.parameters_and_names():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.set_data(initializer(Normal((self.config.initializer_range / math.sqrt(2 * self.config.n_layer))),
                                              p.shape, p.dtype))


class GPT2Model(GPT2PreTrainedModel):
    r"""
    gpt2 Model
    """
    def __init__(self, config):
        """
        Initializes an instance of the GPT2Model class.

        Args:
            self: The GPT2Model instance.
            config:
                An object containing the configuration settings for the model.
                It should include the following attributes:

                - hidden_size (int): The dimensionality of the model's hidden states.
                - vocab_size (int): The size of the vocabulary.
                - max_position_embeddings (int): The maximum length of input sequences.
                - embd_pdrop (float): The dropout probability for the embeddings.
                - num_hidden_layers (int): The number of hidden layers in the model.
                - layer_norm_epsilon (float): The epsilon value to avoid division by zero in layer normalization.
                - add_cross_attention (bool): Whether to add cross attention layers.
                - output_attentions (bool): Whether to output attention tensors.
                - output_hidden_states (bool): Whether to output hidden states.
                - use_cache (bool): Whether to use the cache mechanism.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = Dropout(p=config.embd_pdrop)
        self.h = nn.CellList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm((self.embed_dim,), epsilon=config.layer_norm_epsilon)

        self.add_cross_attention = self.config.add_cross_attention
        self.num_hidden_layers = self.config.num_hidden_layers
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = self.config.use_cache

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

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def construct(
            self,
            input_ids: Tensor,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
    ):
        """
        Constructs the GPT2Model.

        Args:
            self (GPT2Model): The instance of the GPT2Model class.
            input_ids (Tensor): The input tensor of shape (batch_size, sequence_length) containing the input IDs.
            past_key_values (Optional[Tuple[Tuple[Tensor]]]): The optional tuple of past key values. Defaults to None.
            attention_mask (Optional[Tensor]): The optional attention mask tensor of shape (batch_size, sequence_length).
                Defaults to None.
            token_type_ids (Optional[Tensor]): The optional token type IDs tensor of shape (batch_size, sequence_length).
                Defaults to None.
            position_ids (Optional[Tensor]): The optional position IDs tensor of shape (batch_size, sequence_length).
                Defaults to None.
            head_mask (Optional[Tensor]): The optional head mask tensor. Defaults to None.
            inputs_embeds (Optional[Tensor]):
                The optional input embeddings tensor of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            encoder_hidden_states (Optional[Tensor]):
                The optional encoder hidden states tensor of shape (batch_size, encoder_sequence_length, hidden_size).
                Defaults to None.
            encoder_attention_mask (Optional[Tensor]):
                The optional encoder attention mask tensor of shape (batch_size, encoder_sequence_length).
                Defaults to None.

        Returns:
            outputs (None): This method does not return any value.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided.
            ValueError: If neither input_ids nor inputs_embeds are provided.
            ValueError: If batch_size is less than or equal to 0.
        """
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
            position_ids = arange(past_length, input_shape[-1] + past_length, 1, dtype=mindspore.int64)
            position_ids = position_ids.expand_dims(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.astype(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(dtype_to_nptype(self.dtype)).min, self.dtype)

        if self.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        presents = () if self.use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=past_key_values[i],
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=self.use_cache,
            )

            hidden_states = outputs[0]
            if self.use_cache:
                presents = presents + (outputs[1],)

            if self.output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if self.use_cache else 1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if self.use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_attentions:
            outputs += (all_hidden_states, all_self_attentions)
            if self.add_cross_attention:
                outputs += (all_cross_attentions,)
        return outputs


class GPT2LMHeadModel(GPT2PreTrainedModel):
    r"""
    gpt2 LMHead Model
    """
    def __init__(self, config, **kwargs):
        """
        Initializes a GPT2LMHeadModel instance.

        Args:
            self (GPT2LMHeadModel): The GPT2LMHeadModel instance itself.
            config (object): The configuration object containing settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is of an incorrect type.
            ValueError: If 'ignore_index' cannot be popped from kwargs or if an invalid value is provided.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        ignore_index = kwargs.pop('ignore_index', -1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        return the output embeddings layer
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        set the output embeddings layer
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        prepare inputs for generation task
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].expand_dims(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].expand_dims(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].expand_dims(-1)
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
            input_ids: Tensor,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
    ):
        """
        This method 'construct' is defined within the 'GPT2LMHeadModel' class and is responsible for processing
        input data through a transformer model and generating relevant outputs.

        Args:
            self: Represents the instance of the class.
            input_ids (Tensor): The input tensor containing token IDs for the model.
            past_key_values (Optional[Tuple[Tuple[Tensor]]]): A tuple of past key values for fast decoding.
            attention_mask (Optional[Tensor]): An optional attention mask tensor to mask padded tokens.
            token_type_ids (Optional[Tensor]): An optional tensor specifying token types for the input.
            position_ids (Optional[Tensor]): An optional tensor specifying position ids of tokens.
            head_mask (Optional[Tensor]): An optional tensor for masking specific heads in the attention layers.
            inputs_embeds (Optional[Tensor]): An optional tensor representing input embeddings.
            encoder_hidden_states (Optional[Tensor]): An optional tensor containing hidden states from an encoder.
            encoder_attention_mask (Optional[Tensor]): An optional attention mask for the encoder.
            labels (Optional[Tensor]): An optional tensor containing labels for the model training.

        Returns:
            output: A tuple containing the language model logits, and additional transformer outputs.
                If a loss is calculated, it will be included in the output tuple.

        Raises:
            None
        """
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
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits,) + transformer_outputs[1:]
        if loss is not None:
            output = (loss,) + output
        return output

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.astype(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    r"""
    GPT2 Double Heads Model
    """
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        """Initializes a new instance of the GPT2DoubleHeadsModel class.

        Args:
            self: The object instance.
            config: An instance of the GPT2Config class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the embeddings of the obtained output
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Define the embeddings of the output
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        prepare_inputs
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

    def construct(self, input_ids, past_key_values=None, attention_mask=None, token_type_ids=None,
                  position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=None, labels=None, mc_labels=None):
        """
        Constructs the GPT2DoubleHeadsModel.

        Args:
            self (GPT2DoubleHeadsModel): The instance of the GPT2DoubleHeadsModel class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length) containing the token indices.
            past_key_values (Tuple[torch.Tensor], optional):
                The optional tuple of length `num_layers` containing the past key value states for the transformer.
                Defaults to None.
            attention_mask (torch.Tensor, optional):
                The optional attention mask tensor of shape (batch_size, sequence_length)
                containing the mask for padded tokens. Defaults to None.
            token_type_ids (torch.Tensor, optional):
                The optional token type ids tensor of shape (batch_size, sequence_length) containing the token type ids.
                Defaults to None.
            position_ids (torch.Tensor, optional):
                The optional position ids tensor of shape (batch_size, sequence_length) containing the position indices.
                Defaults to None.
            head_mask (torch.Tensor, optional):
                The optional head mask tensor of shape (num_heads,) or (num_layers, num_heads) containing the mask for
                the transformer heads. Defaults to None.
            inputs_embeds (torch.Tensor, optional):
                The optional input embeddings tensor of shape (batch_size, sequence_length, hidden_size) containing
                the input embeddings. Defaults to None.
            mc_token_ids (torch.Tensor, optional):
                The optional multiple choice token ids tensor of shape (batch_size, num_choices) containing the token
                indices for multiple choice questions. Defaults to None.
            labels (torch.Tensor, optional):
                The optional labels tensor of shape (batch_size, sequence_length) containing the token labels for
                language modeling. Defaults to None.
            mc_labels (torch.Tensor, optional):
                The optional multiple choice labels tensor of shape (batch_size,) containing the labels for multiple
                choice classification. Defaults to None.

        Returns:
            Tuple[torch.Tensor]:
                A tuple containing the output logits for language modeling (`lm_logits`), the output logits for
                multiple choice classification (`mc_logits`), and the transformer outputs (`transformer_outputs`).
                The `lm_logits` tensor has shape (batch_size, sequence_length, vocab_size), the `mc_logits` tensor has
                shape (batch_size,), and the `transformer_outputs` is a tuple containing the hidden states of the
                transformer.

        Raises:
            None.
        """
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.shape[-1]), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_loss is not None:
            output = (mc_loss,) + output
        return ((lm_loss,) + output) if lm_loss is not None else output

    @staticmethod
    def _reorder_cache(past, beam_idx):
        '''
        This method '_reorder_cache' belongs to the class 'GPT2DoubleHeadsModel' and is used to reorder the cache based
        on the provided beam index.

        Args:
            past (tuple): A tuple containing the past states of the model.
                Each element in the tuple represents the past states for a specific layer.
                The past states are used for generating the next sequence of tokens.
            beam_idx (torch.Tensor):
                A tensor containing the indices that specify the new order in which the past states should be reordered.
                It is expected to be on the same device as the past states.

        Returns:
            None: This method returns None as the reordered cache is directly updated within the method.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            IndexError: If the beam index is out of range or invalid.
            RuntimeError: If there are any runtime issues with reordering the cache or if the device of the beam index
                does not match the device of the past states.
        '''
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    r"""
    gpt2 For Sequence Classification
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GPT2ForSequenceClassification class.

        Args:
            self (object): The instance of the GPT2ForSequenceClassification class.
            config (object): The configuration object containing settings for the model.
                It should have the following attributes:

                - num_labels (int): The number of labels for classification.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
            self,
            input_ids: Tensor,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
    ):
        """
        Method 'construct' in the class 'GPT2ForSequenceClassification'.

        Args:
            self: The instance of the class.
            input_ids (Tensor): The input tensor IDs representing the input sequence.
            past_key_values (Optional[Tuple[Tuple[Tensor]]]): Optional past key values for the transformer.
            attention_mask (Optional[Tensor]): Optional tensor for masking out tokens.
            token_type_ids (Optional[Tensor]): Optional tensor for token type IDs.
            position_ids (Optional[Tensor]): Optional tensor for position IDs.
            head_mask (Optional[Tensor]): Optional tensor for masking out heads.
            inputs_embeds (Optional[Tensor]): Optional tensor for input embeddings.
            labels (Optional[Tensor]): Optional tensor for target labels.

        Returns:
            None.

        Raises:
            AssertionError: Raised if batch size is greater than 1 and no padding token is defined.
            Warning: Raised if padding tokens are not detected in 'inputs_embeds', which may lead to unexpected results.
        """
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
                sequence_lengths = ops.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    "%s will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding "
                    "tokens in conjunction with `inputs_embeds.`", self.__class__.__name__)

        pooled_logits = logits[:, sequence_lengths]

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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        output = (pooled_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output


class GPT2ForTokenClassification(GPT2PreTrainedModel):
    r"""
    GPT2 For Token Classification
    """
    def __init__(self, config):
        """
        Initializes an instance of the GPT2ForTokenClassification class.

        Args:
            self: The object instance.
            config:
                An instance of the GPT2Config class containing the model configuration settings.

                - Type: GPT2Config
                - Purpose: Specifies the configuration parameters for the GPT-2 model.
                - Restrictions: None
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None,
                  position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        Args:
            self (object): The instance of the class.
            input_ids (Tensor, optional): The input token IDs. Default is None.
            past_key_values (tuple, optional):
                Tuple of length 2 containing tensors of past key and value states for fast autoregressive decoding.
                Default is None.
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
                Default is None.
            token_type_ids (Tensor, optional): Segment token indices to differentiate two sequences. Default is None.
            position_ids (Tensor, optional): Indices of positions in the input sequences. Default is None.
            head_mask (Tensor, optional): Mask to nullify selected heads of the self-attention modules. Default is None.
            inputs_embeds (Tensor, optional): Optional input embeddings overrides input_ids. Default is None.
            labels (Tensor, optional): Labels for computing the token classification loss. Default is None.
        
        Returns:
            tuple: A tuple containing the loss (if labels are provided) and the output logits and transformer outputs.
        
        Raises:
            ValueError: If the input_ids and inputs_embeds are both set.
            ValueError: If the length of past_key_values tuple is not 2.
            ValueError: If the length of transformer_outputs is not as expected.
        """
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + transformer_outputs[2:]
        return ((loss,) + output) if loss is not None else output
