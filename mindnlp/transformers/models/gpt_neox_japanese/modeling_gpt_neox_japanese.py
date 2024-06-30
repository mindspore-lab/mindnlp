# coding=utf-8
# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore GPTNeoX model."""

from typing import Optional, Tuple, Union

import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer

from mindnlp.utils import get_default_dtype, logging

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "abeja/gpt-neox-japanese-2.7b"
_CONFIG_FOR_DOC = "GPTNeoXJapaneseConfig"


class GPTNeoXJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXJapaneseConfig
    base_model_prefix = "gpt_neox_japanese"
    _no_split_modules = ["GPTNeoXJapaneseLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    shape=cell.weight.shape,
                    dtype=cell.weight.dtype,
                )
            )
            if cell.bias:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.padding_idx:
                cell.weight[cell.padding_idx] = 0
        elif isinstance(cell, nn.LayerNorm):
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
            cell.weight.set_data(
                initializer("ones", cell.weight.shape, cell.weight.dtype)
            )


class GPTNeoXJapaneseAttention(nn.Cell):
    def __init__(self, config, use_bias=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads

        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            config.max_position_embeddings,
            base=config.rotary_emb_base,
        )
        self.max_positions = config.max_position_embeddings
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.norm_factor = ops.sqrt(
            mindspore.tensor(self.head_size, dtype=mindspore.float32)
        )

        self.query_key_value = nn.Dense(
            config.hidden_size, 3 * config.hidden_size, has_bias=False
        )
        self.dense = nn.Dense(config.hidden_size, config.hidden_size, has_bias=False)
        # Activate bias if the last layer
        self.use_bias = use_bias
        self.dense_bias = (
            mindspore.Parameter(ops.zeros(config.hidden_size)) if use_bias else None
        )

    def construct(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None and ops.numel(layer_past[0]) > 0

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.reshape(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = ops.cat((query, query_pass), axis=-1)
        key = ops.cat((key, key_pass), axis=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = ops.cat((past_key, key), axis=-2)
            value = ops.cat((past_value, value), axis=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        # Reshape outputs
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs, self.dense_bias

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.shape[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.reshape(
            tensor.shape[0], tensor.shape[1], num_attention_heads * attn_head_size
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _create_causal_mask(self, key_length, query_length):
        causal_mask = ops.tril(
            ops.ones(
                (self.max_positions, self.max_positions), dtype=mindspore.bool_
            ).reshape(1, 1, self.max_positions, self.max_positions)
        )
        return causal_mask[:, :, key_length - query_length : key_length, :key_length]

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.shape
        key_length = key.shape[-2]

        causal_mask = self._create_causal_mask(key_length, query_length)

        query = query.reshape(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.reshape(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = ops.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
        )
        attn_scores = ops.baddbmm(
            attn_scores,
            query,
            key.swapaxes(1, 2),
            beta=0,
            alpha=(1.0 / float(self.norm_factor)),
        )
        attn_scores = attn_scores.reshape(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = np.finfo(mindspore.dtype_to_nptype(attn_scores.dtype)).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = mindspore.tensor(mask_value, dtype=attn_scores.dtype)
        attn_scores = ops.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = ops.softmax(attn_scores, axis=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)
        return attn_output, attn_weights


# Copied from transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXRotaryEmbedding with GPTNeoXRotaryEmbedding->RotaryEmbedding
class RotaryEmbedding(nn.Cell):
    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding.__init__
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim)
        )
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            dtype=get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(
            self.inv_freq
        )

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def bias_dropout_add(
    x: mindspore.Tensor,
    bias: mindspore.Tensor,
    residual: Optional[mindspore.Tensor],
    prob: float,
    training: bool,
) -> mindspore.Tensor:
    """add bias to x, apply dropout and residual connection

    Args:
        x (Tensor): main path of output
        bias (Tensor): None or attn_bias of the last attention layer
        residual (Optional[Tensor]): residual value
        prob (float): dropout probability
        training (bool): whether in training mode or not

    Returns:
        Tensor: Dropout(p=x + bias) + residual
    """
    if bias is not None:
        x = x + bias
    out = ops.dropout(input=x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


class GPTNeoXJapaneseMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        intermediate_size = int(config.hidden_size * config.intermediate_multiple_size)
        self.dense_h_to_4h = nn.Dense(
            config.hidden_size, intermediate_size, has_bias=False
        )
        # Project back to h.
        self.dense_4h_to_h = nn.Dense(
            intermediate_size, config.hidden_size, has_bias=False
        )
        self.act = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXJapaneseLayer(nn.Cell):
    def __init__(self, config, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.input_layernorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )
        # activate bias only last layer
        self.attention = GPTNeoXJapaneseAttention(
            config=config, use_bias=layer_number == config.num_hidden_layers - 1
        )
        self.mlp = GPTNeoXJapaneseMLP(config)
        self.hidden_dropout = config.hidden_dropout

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)
        attention_layer_outputs, attn_bias = self.attention(
            ln_out,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: a, present, (attentions)
        outputs = attention_layer_outputs[1:]

        # attn_output = (atten_output + bias) + residual
        attn_output = bias_dropout_add(
            x=attn_output,
            bias=attn_bias.expand_as(residual) if attn_bias is not None else attn_bias,
            residual=residual,
            prob=self.hidden_dropout,
            training=self.training,
        )
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))

        # attn_output = (mlp_output + mlp_bias) + atten_output
        attn_output = bias_dropout_add(
            x=mlp_output,
            bias=None,
            residual=attn_output,
            prob=self.hidden_dropout,
            training=self.training,
        )

        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPTNeoXJapaneseModel(GPTNeoXJapanesePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.CellList(
            [
                GPTNeoXJapaneseLayer(config=config, layer_number=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors
            of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).

        Returns:
            `Union[Tuple, BaseModelOutputWithPast]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, GPTNeoXJapaneseModel
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
            >>> model = GPTNeoXJapaneseModel.from_pretrained("abeja/gpt-neox-japanese-2.7b")
            ...
            >>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        # Attention mask.
        if attention_mask is not None:
            if not batch_size > 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.reshape(batch_size, -1)
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
            attention_mask = (1.0 - attention_mask) * mindspore.tensor(np.finfo(
                mindspore.dtype_to_nptype(attention_mask.dtype)
            ).min)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoXJapaneseForCausalLM(GPTNeoXJapanesePreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.gpt_neox_japanese = GPTNeoXJapaneseModel(config)
        self.embed_out = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
                when `config.use_cache=True`):

                - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
                only required when the model is used as a decoder in a Sequence to Sequence model.
                - Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
                `past_key_values` input) to speed up sequential decoding.
                - If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).

        Returns:
            `Union[Tuple, CausalLMOutputWithPast]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseConfig
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
            >>> config = GPTNeoXJapaneseConfig.from_pretrained("abeja/gpt-neox-japanese-2.7b")
            >>> config.is_decoder = True
            >>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", config=config)
            ...
            >>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.logits
            ```
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.gpt_neox_japanese(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism

            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(
                shift_logits.reshape(
                    -1,
                    shift_logits.shape[-1],
                ),
                labels.reshape(-1),
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


__all__ = [
    "GPTNeoXJapaneseForCausalLM",
    "GPTNeoXJapaneseLayer",
    "GPTNeoXJapaneseModel",
    "GPTNeoXJapanesePreTrainedModel",
]
