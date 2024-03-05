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
from mindspore import ops, nn, Parameter, Tensor
from mindspore import log as logger
from mindspore.common.initializer import initializer, Normal



from mindnlp.transformers.generation.logits_process import (
    BarkEosPrioritizerLogitsProcessor,
    SuppressTokensLogitsProcessor,
    AlternatingCodebooksLogitsProcessor
)
from mindnlp.transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from ...modeling_utils import PreTrainedModel
from ..auto import AutoModel
from .configuration_bark import (
    BarkConfig,
    BarkSubModelConfig,
    BarkSemanticConfig,
    BarkCoarseConfig,
    BarkFineConfig
)

from .generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

from ...modeling_outputs import (
    CausalLMOutputWithPast,
    MaskedLMOutput,
)

BARK_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bark",
    "bark_small"
}

__all__ = [
    "BarkModel",
    "BarkSelfAttention",
    "BarkMLP",
    "BarkLayerNorm",
    "BarkBlock",
    "BarkCausalModel",
    "BarkSemanticModel",
    "BarkCoarseModel",
    "BarkFineModel"
]

class BarkSelfAttention(nn.Cell):
    r"""
    BarkSelfAttention
    """
    def __init__(self, config, is_causal=False):
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
            self.bias = Parameter(Tensor(np.tril(np.ones((block_size, block_size))).reshape(
                (1, 1, block_size, block_size)
            ), mindspore.bool_), requires_grad=False)

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
        # tensor = tensor.contiguous()
        tensor = tensor.view(tensor.shape[:-2] + (num_heads * attn_head_size,))
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # unlike GPTNeo's SelfAttention, divide by the square root of the dimension of the query and the key
        attn_weights = ops.matmul(query, key.swapaxes(-1, -2)) * (1.0 / math.sqrt(key.shape[-1]))

        if self.is_causal:
            query_length, key_length = query.shape[-2], key.shape[-2]
            # fill the upper left part of the attention weights with inf
            attn_weights = attn_weights.masked_fill(
                self.bias[:, :, key_length - query_length : key_length, :key_length] == 0,
                mindspore.tensor(np.finfo(np.float32).min),
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

class BarkSelfFlashAttention2(BarkSelfAttention):
    """
    Bark flash attention module. This module inherits from `BarkSelfAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = True

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim - (batch, seq_length, head, head_features)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # re-assemble all head outputs side by side
        # (batch, seq_len, num_heads, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))
        return tensor

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
    ):
        r"""
        construct
        """
        _, query_len, _ = hidden_states.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if past_key_values is not None:
            # (batch, head, seq_length, head_features) -> (batch, seq_length, head, head_features)
            past_key = past_key_values[0].swapaxes(1, 2)
            past_value = past_key_values[1].swapaxes(1, 2)
            # and merge on seq_length
            key = ops.cat((past_key, key), axis=1)
            value = ops.cat((past_value, value), axis=1)

        if use_cache is True:
            #  (batch, head, seq_length, head_features)
            present = (key.swapaxes(1, 2), value.swapaxes(1, 2))
        else:
            present = None

        attn_output = self._flash_attention_forward(query, key, value, attention_mask, query_len, dropout=self.dropout)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            attn_weights = None
            outputs += (attn_weights,)

        return outputs

    # # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    # def _flash_attention_forward(
    #     self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    # ):
    #     """
    #     Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    #     first unpad the input, then computes the attention scores and pad the final attention scores.

    #     Args:
    #         query_states (`torch.Tensor`):
    #             Input query states to be passed to Flash Attention API
    #         key_states (`torch.Tensor`):
    #             Input key states to be passed to Flash Attention API
    #         value_states (`torch.Tensor`):
    #             Input value states to be passed to Flash Attention API
    #         attention_mask (`torch.Tensor`):
    #             The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
    #             position of padding tokens and 1 for the position of non-padding tokens.
    #         dropout (`int`, *optional*):
    #             Attention dropout
    #         softmax_scale (`float`, *optional*):
    #             The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    #     """
    #     if not self._flash_attn_uses_top_left_mask:
    #         causal = self.is_causal
    #     else:
    #         # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
    #         causal = self.is_causal and query_length != 1

    #     # Contains at least one padding token in the sequence
    #     if attention_mask is not None:
    #         batch_size = query_states.shape[0]
    #         query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
    #             query_states, key_states, value_states, attention_mask, query_length
    #         )

    #         cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    #         max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    #         attn_output_unpad = flash_attn_varlen_func(
    #             query_states,
    #             key_states,
    #             value_states,
    #             cu_seqlens_q=cu_seqlens_q,
    #             cu_seqlens_k=cu_seqlens_k,
    #             max_seqlen_q=max_seqlen_in_batch_q,
    #             max_seqlen_k=max_seqlen_in_batch_k,
    #             dropout_p=dropout,
    #             softmax_scale=softmax_scale,
    #             causal=causal,
    #         )

    #         attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    #     else:
    #         attn_output = flash_attn_func(
    #             query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
    #         )

    #     return attn_output

    # # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    #     indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    #     batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    #     key_layer = index_first_axis(
    #         key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    #     )
    #     value_layer = index_first_axis(
    #         value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    #     )
    #     if query_length == kv_seq_len:
    #         query_layer = index_first_axis(
    #             query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
    #         )
    #         cu_seqlens_q = cu_seqlens_k
    #         max_seqlen_in_batch_q = max_seqlen_in_batch_k
    #         indices_q = indices_k
    #     elif query_length == 1:
    #         max_seqlen_in_batch_q = 1
    #         cu_seqlens_q = torch.arange(
    #             batch_size + 1, dtype=torch.int32, device=query_layer.device
    #         )  # There is a memcpy here, that is very bad.
    #         indices_q = cu_seqlens_q[:-1]
    #         query_layer = query_layer.squeeze(1)
    #     else:
    #         # The -q_len: slice assumes left padding.
    #         attention_mask = attention_mask[:, -query_length:]
    #         query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    #     return (
    #         query_layer,
    #         key_layer,
    #         value_layer,
    #         indices_q,
    #         (cu_seqlens_q, cu_seqlens_k),
    #         (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    #     )


BARK_ATTENTION_CLASSES = {
    "eager": BarkSelfAttention,
    "flash_attention_2": BarkSelfFlashAttention2,
}

class BarkLayerNorm(nn.Cell):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.weight = mindspore.Parameter(ops.ones(hidden_size))
        self.bias = mindspore.Parameter(ops.zeros(hidden_size)) if bias else None


    def construct(self, inputs):
        layer_norm = nn.LayerNorm(self.weight.shape, gamma_init=self.weight, beta_init=self.bias if self.bias is not None else 'zeros', epsilon=1e-5)
        return layer_norm(inputs)

class BarkMLP(nn.Cell):
    r"""
    BarkMLP
    """
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Dense(config.hidden_size, 4 * config.hidden_size, has_bias=config.bias)
        self.out_proj = nn.Dense(4 * config.hidden_size, config.hidden_size, has_bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)
        self.gelu = nn.GELU()

    def construct(self, hidden_states):
        r"""
        BarkMLP construct method
        """
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BarkBlock(nn.Cell):
    r"""
    BarkBlock
    """
    def __init__(self, config, is_causal=False):
        super().__init__()

        if is_causal:
            # if causal, uses handmade LayerNorm, so that the layerNorm bias is optional
            # this handmade layerNorm is used to stick with Bark choice of leaving optional bias in
            # AutoRegressive models (corresponding to the "Text" and the "Coarse" modules)
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm([config.hidden_size])
            self.layernorm_2 = nn.LayerNorm([config.hidden_size])

        self.attn = BarkSelfAttention(config, is_causal=is_causal)

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
        r"""
        BarkBlock construct method
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
    _supports_flash_attn_2 = False
    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, (nn.Dense,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(mean=0.0, sigma=self.config.initializer_range),
                                            shape=cell.weight.shape,
                                            dtype=cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            embedding_table = initializer(Normal(self.config.initializer_range),
                                        cell.weight.shape,
                                        cell.weight.dtype)
            if cell.padding_idx is not None:
                embedding_table.data[cell.padding_idx] = 0
            cell.weight.set_data(embedding_table)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            # cell.gammaParameter.data.zero()
            # cell.beta.data.fill(1.0)


    def _set_gradient_checkpointing(self, cell, value=False):
        if isinstance(cell, (BarkCausalModel, BarkFineModel, BarkModel)):
            cell.gradient_checkpointing = value

# GPT2-like autoregressive model
class BarkCausalModel(BarkPreTrainedModel):
    r"""
    BarkCausalModel
    """
    config_class = BarkSubModelConfig

    def __init__(self, config):
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
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
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
            # position_ids = position_ids
            position_ids.masked_fill(attention_mask == 0, 1)
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
        r"""
        construct
        """
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
        if input_ids is not None:
            input_embeds = self.input_embeds_layer(input_ids)  # token embeddings of shape (b, t, n_embd)
        if input_embeds is not None:
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
            position_ids = position_ids.expand_dims(0).view(-1, seq_length)
            #position_ids = position_ids.unsqueeze(0)  # shape (1, seq_length)

        position_embeds = self.position_embeds_layer(position_ids)  # position embeddings of shape (1, t, n_embd)

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            else:
                attention_mask = attention_mask.view(batch_size, -1)
                # [bsz, to_seq_length] -> [bsz, 1, 1, to_seq_length]
                # from_seq_length is 1 to easily broadcast
                attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)

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

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
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
    r"""
    BarkSemanticModel
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
            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids, i.e tokenized input sentences. Will be truncated up to
                semantic_generation_config.max_input_semantic_length tokens. Note that the output audios will be as
                long as the longest generation among the batch.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
            attention_mask (`Optional[torch.Tensor]`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        Returns:
            torch.LongTensor: Output semantic tokens.
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
                [semantic_generation_config.semantic_pad_token] * max_input_semantic_length, dtype=mindspore.int64
            )

        semantic_history = ops.repeat_interleave(semantic_history[None], batch_size, axis=0)

        infer_array = mindspore.Tensor(
            [[semantic_generation_config.semantic_infer_token]] * batch_size, dtype=mindspore.int64
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
    r"""
    BarkCoarseModel
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
            history_prompt (`Optional[Dict[str,torch.Tensor]]`):
                Optional `Bark` speaker prompt.
        Returns: Returns:
            `tuple(torch.FloatTensor)`:
            - **x_semantic_history** (`torch.FloatTensor` -- Processed semantic speaker prompt.
            - **x_coarse_history** (`torch.FloatTensor`) -- Processed coarse speaker prompt.
        """
        if history_prompt is not None:
            x_semantic_history = ops.repeat_interleave(history_prompt["semantic_prompt"][None], batch_size, axis=0)
            # clone to avoid modifying history_prompt.coarse_prompt
            x_coarse_history = history_prompt["coarse_prompt"]

            # offset x_coarse_history
            if codebook_size is not None:
                for i in range(1, x_coarse_history.shape[0]):
                    # offset
                    x_coarse_history[i, :] += codebook_size * i

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

            x_semantic_history = x_semantic_history[:, -n_semantic_hist_provided:].astype(mindspore.int64)
            x_coarse_history = x_coarse_history[:, -n_coarse_hist_provided:]
            # bit of a hack for time alignment (sounds better) - from Bark original implementation
            x_coarse_history = x_coarse_history[:, :-2].astype(mindspore.int64)

        else:
            # shape: (batch_size, 0)
            class Test:
                r"""
                help to decorate zero dirmation martix
                """
                def __init__(self) -> None:
                    self.__enable_zero_dim__ = True
            x_semantic_history = mindspore.Tensor(dtype=mindspore.int64, shape=(batch_size, 0), init= Test())
            x_coarse_history = mindspore.Tensor(dtype=mindspore.int64, shape=(batch_size, 0), init= Test())

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
    ) -> mindspore.Tensor:
        """
        Generates coarse acoustics tokens from input text semantic tokens and an additional optional `Bark` speaker
        prompt.

        Args:
            semantic_output (`torch.Tensor` of shape (batch_size, seq_len), *optional*):
                Input text semantic ids, i.e the output of `BarkSemanticModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
        Returns:
            torch.LongTensor: Output coarse acoustics tokens.
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
        semantic_output.masked_fill(
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
        max_generated_len = ops.max(output_lengths)[0]

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
            input_coarse = semantic_output[:, int(np.max([0, semantic_idx - max_semantic_history])):]
            input_coarse = input_coarse[:, :max_coarse_input_length]
            input_coarse = ops.pad(
                input_coarse,
                (0, max_coarse_input_length - input_coarse.shape[-1]),
                "constant",
                coarse_generation_config.coarse_semantic_pad_token,
            )
            x_coarse_t = x_coarse if x_coarse.shape[1] == 0 else x_coarse[:, int(-max_coarse_history):]
            input_coarse = ops.hstack(
                [
                    input_coarse,
                    mindspore.Tensor([[coarse_generation_config.coarse_infer_token]] * batch_size),
                    x_coarse_t,
                ]
            )

            alternatinglogitsprocessor = AlternatingCodebooksLogitsProcessor(
                input_coarse.shape[1],
                semantic_generation_config.semantic_vocab_size,
                codebook_size,
            )
            output_coarse = super().generate(
                input_coarse,
                logits_processor=[alternatinglogitsprocessor],
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
    r"""
    BarkFineModel
    """
    base_model_prefix = "fine_acoustics"
    config_class = BarkFineConfig
    main_input_name = "codebook_idx"

    def __init__(self, config):
        # non-causal gpt-like model with one embedding layer and one lm_head for each codebook of Encodec
        super().__init__(config)
        self.config = config
        self._tied_weights_keys = []

        # initialize a modified non causal GPT-like model
        # note that for there is one embedding layer and one lm_head for each codebook of Encodec
        self.input_embeds_layers = nn.CellList(
            [nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)]
        )
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        self.drop = nn.Dropout(p=config.dropout)

        self.layers = nn.CellList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])

        self.layernorm_final = nn.LayerNorm([config.hidden_size])

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
        # one embedding layers for each codebook
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        # one embedding layers for each codebook
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        # one lm_head for each codebook
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        # one lm_head for each codebook
        self.lm_heads = new_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
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
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def tie_weights(self):
        """
        Tie the weights between the input embeddings list and the output embeddings list.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            self._tied_weights_keys = []
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()

            for i in range(self.config.n_codes_total - self.config.n_codes_given):
                # self.input_embeds_layers[i + 1].weight = self.lm_heads[i].weight
                self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
                self._tied_weights_keys.append(f"lm_heads.{i}.weight")

        for cell in self.cells():
            if hasattr(cell, "_tie_weights"):
                cell._tie_weights()

    def construct(
        self,
        codebook_idx: int = 4,  # an additionnal idx corresponding to the id of the codebook that will be predicted
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
        r"""
        construct
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
        # device = input_ids.device if input_ids is not None else input_embeds.device
        if position_ids is None:
            position_ids = ops.arange(0, seq_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)  # shape (1, seq_length)
        position_embeds = self.position_embeds_layer(position_ids)  # position embeddings of shape (1, t, n_embd)
        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            else:
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
            coarse_output (`torch.Tensor` of shape (batch_size, seq_len)):
                Input coarse acoustics ids, i.e the output of `BarkCoarseModel.generate`.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            coarse_generation_config (`BarkCoarseGenerationConfig`):
                Generation config indicating how to generate the coarse tokens.
            fine_generation_config (`BarkFineGenerationConfig`):
                Generation config indicating how to generate the fine tokens.
            codebook_size (`int`, *optional*, defaults to 1024):
                Codebook channel size, i.e. the size of the output vocabulary per codebook channel.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
        Returns:
            torch.LongTensor: Output fine acoustics tokens.
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
            x_fine_history = ops.repeat_interleave(history_prompt["fine_prompt"].T[None], batch_size, axis=0)
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
            fine_input = ops.cat([x_fine_history[:, -max_fine_history_length:, :].astype(mindspore.int64), fine_input], axis=1)

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
    r"""
    BarkModel
    """
    config_class = BarkConfig

    def __init__(self, config):
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
            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the
                longest generation among the batch.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.
            kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *semantic_*, *coarse_*, *fine_* prefix, they will be input for the `generate` method of the
                semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for all sub-models except one.
        Returns:
            torch.LongTensor: Output generated audio.

        Example:

        ```python
        >>> from transformers import AutoProcessor, BarkModel

        >>> processor = AutoProcessor.from_pretrained("suno/bark-small")
        >>> model = BarkModel.from_pretrained("suno/bark-small")

        >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
        >>> voice_preset = "v2/en_speaker_6"

        >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)

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
        # 4. Decode the output and generate audio array
        audio = self.codec_decode(output, output_lengths=output_lengths)
        if return_output_lengths:
            output_lengths = [len(sample) for sample in audio]
            audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
            return audio, output_lengths

        return audio

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[any] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        hard_check_only: bool = False,
    ):
        """
        `_check_and_enable_flash_attn_2` originally don't expand flash attention enabling to the model
        sub-configurations. We override the original method to make sure that Bark sub-models are using Flash Attention
        if necessary.

        If you don't know about Flash Attention, check out the official repository of flash attention:
        https://github.com/Dao-AILab/flash-attention

        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this
        specific section of the documentation to learn more about it:
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models

        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in
        half precision and not ran on CPU.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flash_attention_2" so that the model
        can initialize the correct attention module
        """
        config = super()._check_and_enable_flash_attn_2(
            config, torch_dtype, device_map, hard_check_only=hard_check_only
        )

        config.semantic_config._attn_implementation = config._attn_implementation
        config.coarse_acoustics_config._attn_implementation = config._attn_implementation
        config.fine_acoustics_config._attn_implementation = config._attn_implementation
        return config
