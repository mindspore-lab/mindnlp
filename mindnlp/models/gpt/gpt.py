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
"""MindNLP gpt model"""
import math
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter
from mindnlp._legacy.functional import split, tril
from mindnlp._legacy.nn import Dropout
from ..utils.utils import Conv1D, prune_conv1d_layer, find_pruneable_heads_and_indices
from ..utils.activations import ACT2FN
from ..utils import logging

logger = logging.get_logger(__name__)

class GPTMLP(nn.Cell):
    r"""
    GPT MLP
	"""

    def __init__(self, n_state, config):
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = ACT2FN[config.afn]
        self.dropout = Dropout(p=config.resid_pdrop)

    def construct(self, x):
        hidden_states1 = self.c_fc(x)
        hidden_states2 = self.act(hidden_states1)
        hidden_states3 = self.c_proj(hidden_states2)
        outputs = self.dropout(hidden_states3)
        return outputs


class GPTAttention(nn.Cell):
    r"""
    GPT Attention
    """

    def __init__(self, config, scale=False):
        super().__init__()
        n_state = config.n_embd
        n_positions = config.n_positions
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")
        bias_tensor = tril(ops.ones((n_positions, n_positions), mindspore.int32))
        self.bias = Parameter(bias_tensor.view(1, 1, n_positions, n_positions), requires_grad=False)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_proj = Conv1D(n_state, n_state)
        self.attn_dropout = Dropout(p=config.attn_pdrop)
        self.resid_dropout = Dropout(p=config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes heads of the model.
        """
        if len(heads) == 0:
            return
        head_size = self.split_size//self.n_head
        heads, index = find_pruneable_heads_and_indices(heads, self.n_head, head_size, self.pruned_heads)
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, axis=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, axis=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, output_attentions=False):
        attn_weights = ops.matmul(query, key)
        if self.scale:
            attn_weights = attn_weights / math.sqrt(value.shape[-1])

        attn_bias = self.bias[:, :, : attn_weights.shape[-2], : attn_weights.shape[-1]]
        attn_weights = attn_weights * attn_bias + -1e4 * (1 - attn_bias)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        outputs = [ops.matmul(attn_weights, value)]
        if output_attentions:
            outputs.append(attn_weights)
        return outputs

    def merge_heads(self, tensor):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.transpose(0, 2, 1, 3)
        new_tensor_shape = tensor.shape[:-2] + (tensor.shape[-2] * tensor.shape[-1],)
        return tensor.view(*new_tensor_shape)

    def split_heads(self, tensor, tensor_transpose=False):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_tensor_shape = tensor.shape[:-1] + (self.n_head, tensor.shape[-1] // self.n_head)
        tensor = tensor.view(*new_tensor_shape)
        if tensor_transpose:
            return tensor.transpose(0, 2, 3, 1)
        return tensor.transpose(0, 2, 1, 3)

    def construct(self, input_states, attention_mask=None, head_mask=None, output_attentions=False):
        c_states = self.c_attn(input_states)
        query, key, value = split(c_states, self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, tensor_transpose=True)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)

        output = attn_outputs[0]
        output = self.merge_heads(output)
        output = self.c_proj(output)
        output = self.resid_dropout(output)
        outputs = [output] + attn_outputs[1:]
        return outputs


class GPTBlock(nn.Cell):
    r"""
    GPT Block
    """

    def __init__(self, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        self.attn = GPTAttention(config, scale)
        self.ln_1 = nn.LayerNorm(normalized_shape=[hidden_size], epsilon=config.layer_norm_epsilon)
        self.mlp = GPTMLP(4 * hidden_size, config)
        self.ln_2 = nn.LayerNorm(normalized_shape=[hidden_size], epsilon=config.layer_norm_epsilon)

    def construct(self, input_states, attention_mask=None, head_mask=None, output_attentions=False):
        residual_1 = input_states
        attn_outputs = self.attn(input_states, attention_mask, head_mask, output_attentions,)
        attn_output = attn_outputs[0]
        hidden_states = self.ln_1(residual_1 + attn_output)
        residual_2 = hidden_states
        feed_forward_hidden_states = self.mlp(hidden_states)
        output_hidden_states = self.ln_2(residual_2 + feed_forward_hidden_states)
        outputs = [output_hidden_states] + attn_outputs[1:]
        return outputs


class GPTPreTrainedModel(nn.Cell):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "transformer"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Dense, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        """
        -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        """
        if head_mask.dim() == 1:
            head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


class GPTModel(GPTPreTrainedModel):
    """
    The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.block_list = nn.CellList([GPTBlock(config, scale=True) for _ in range(config.n_layer)])
        self.position_ids = Parameter(ops.arange(config.n_positions), requires_grad=False)

    def get_input_embeddings(self):
        """
        return the input embeddings layer
        """
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        """
        set the input embeddings layer
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
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min,
                                                             self.dtype)

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
        for i, block in enumerate(self.block_list):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, attention_mask, head_mask[i], output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
