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
# pylint: disable=C0103
# pylint: disable=C0415

"""
T5 model
"""

import os
import logging
import math
import copy
import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Parameter, Tensor

from mindnlp._legacy.nn import Dropout
from mindnlp._legacy.functional import arange
from mindnlp.configs import MINDNLP_MODEL_URL_BASE
from ..utils.activations import ACT2FN
from ...abc import PreTrainedModel

from .t5_config import T5Config, T5_SUPPORT_LIST


__all__ = ['T5Attention', 'T5DenseActDense', 'T5DenseGatedActDense', 'T5EncoderModel',
           'T5ForConditionalGeneration', 'T5LayerCrossAttention', 'T5Stack', 'T5LayerSelfAttention',
           'T5LayerNorm', 'T5Model', 'T5LayerFF', 'T5Block', 'T5PreTrainedModel']

PRETRAINED_MODEL_ARCHIVE_MAP = {
    model: MINDNLP_MODEL_URL_BASE.format('t5', model) for model in T5_SUPPORT_LIST
}

def torch_to_mindspore(pth_file, **kwargs):
    """torch to mindspore."""
    prefix = kwargs.get("prefix", "")

    try:
        import torch
    except Exception as exc:
        raise ImportError("'import torch' failed, please install torch by "
                          "`pip install torch` or instructions from 'https://pytorch.org'") \
        from exc

    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'shared.weight' in k:
            k = k.replace('shared.weight', 'decoder.embed_tokens.embedding_table')
        if 'relative_attention_bias.weight' in k:
            k = k.replace('relative_attention_bias.weight', 'relative_attention_bias.embedding_table')
        if prefix:
            k = prefix + "." + k
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('pytorch_model.bin','mindspore.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except Exception as exc:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.') \
            from exc

    return ms_ckpt_path

class T5LayerNorm(nn.Cell):
    """T5LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size, mindspore.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):

        variance = hidden_states.astype(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states / ops.sqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16]:
            hidden_states = hidden_states.astype(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Cell):
    """T5DenseActDense"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.wo.weight.dtype not in (hidden_states.dtype, mindspore.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Cell):
    """T5DenseGatedActDense"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wi_1 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        if self.wo.weight.dtype not in (hidden_states.dtype, mindspore.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5LayerFF(nn.Cell):
    """T5LayerFF"""
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = Dropout(p=config.dropout_rate)

    def construct(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Cell):
    """T5Attention"""
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.k = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.v = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.o = nn.Dense(self.inner_dim, self.d_model, has_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = 0 - \
                ops.minimum(relative_position, ops.zeros(relative_position.shape)).astype(mindspore.int64)
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.astype(mindspore.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.fill(relative_position_if_large.dtype, \
                                                 relative_position_if_large.shape, num_buckets - 1)
        )
        # relative_buckets += ops.where(is_small, relative_position\
        # , relative_position_if_large) # mindspore 2.0
        relative_buckets += ops.select(is_small.astype(mindspore.bool_), \
        relative_position, relative_position_if_large) # mindspore 1.10
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).expand_dims(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def construct(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).swapaxes(1, 2)

        def unshape(states):
            """reshape"""
            return states.swapaxes(1, 2).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], axis=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.swapaxes(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1], mindspore.float32)
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.astype(mindspore.float32), axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        if self.training:
            attn_weights = ops.dropout(
                attn_weights, p=self.dropout
            )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5LayerSelfAttention(nn.Cell):
    """T5LayerSelfAttention"""
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerCrossAttention(nn.Cell):
    """T5LayerCrossAttention"""
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Cell):
    """T5Block"""
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.CellList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        # return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logging.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = Tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
                clamp_value = Tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = Tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
        # hidden-states, present_key_value_states, (self-attention position bias),
        # (self-attention weights), (cross-attention position bias),(cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    base_model_prefix = "transformer"
    convert_torch_to_mindspore = torch_to_mindspore
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    # TODO
    def get_input_embeddings(self):
        pass

    #TODO
    def get_position_embeddings(self):
        pass

    #TODO
    def resize_position_embeddings(self):
        pass

    #TODO
    def set_input_embeddings(self):
        pass

    #TODO
    def post_init(self):
        pass

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = ops.zeros(input_ids.shape, input_ids.dtype)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = ops.masked_fill(shifted_input_ids, shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    """T5Stack"""
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.CellList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length), mindspore.float32)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones(
                (batch_size, encoder_seq_length), mindspore.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), \
            # (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        output = (hidden_states,)+(present_key_value_state,)+\
            (all_hidden_states,)+(all_attentions,)+(all_cross_attentions,)
        return output


class T5Model(T5PreTrainedModel):
    """T5Model"""
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def get_decoder(self):
        """get decoder"""
        return self.decoder

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs + encoder_outputs

class T5ForConditionalGeneration(T5PreTrainedModel):
    """T5ForConditionalGeneration"""
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """set input embeddings"""
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        """set output embeddings"""
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        """get output embeddings"""
        return self.lm_head

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def get_decoder(self):
        """get decoder"""
        return self.decoder

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            # TODO(thom): Add z_loss

        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
    ):
        """prepare inputs for generation"""
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """prepare decoder input ids from labels"""
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logging.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class T5EncoderModel(T5PreTrainedModel):
    """T5EncoderModel"""
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        """get encoder"""
        return self.encoder
    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
