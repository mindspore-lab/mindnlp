# coding=utf-8
# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teamindspore. All rights reserved.
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


""" MindSpore CodeGen model."""

from typing import Optional, Tuple, Any, Union

import mindspore
import numpy as np
from mindspore import nn, Tensor, Parameter
from mindspore import ops

from mindnlp.models.utils import logging
from mindnlp.models.utils.activations import ACT2FN

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/codegen-2B-mono"
_CONFIG_FOR_DOC = "CodeGenConfig"


#
def fixed_pos_embedding(tensor, seq_dim=1, seq_len=None):
    """
    fixed_pos_embedding
    """
    axis = tensor.shape[-1]
    if seq_len is None:
        seq_len = tensor.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (np.arange(0, axis, 2) / axis))
    sinusoid_inp = (
        np.einsum("i,j->ij", np.arange(seq_len, dtype=np.float32), inv_freq).astype(np.float32)
    )
    return Tensor(np.sin(sinusoid_inp)), Tensor(np.cos(sinusoid_inp))


def rotate_every_two(tensor):
    """
    rotate_every_two
    """
    tensor1 = tensor[:, :, :, ::2]
    tensor2 = tensor[:, :, :, 1::2]
    tensor = ops.stack((-tensor2, tensor1), axis=-1)
    return tensor.flatten(order='C', start_dim=-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(tensor):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = tensor.shape[0]
    tensor = tensor.view(-1, 1)  # flatten the matrix
    tensor = ops.tile(tensor, (1, 2))  # repeat all elements into the 2nd dimension
    tensor = tensor.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return tensor


def apply_rotary_pos_emb(tensor, sincos, offset=0):
    """
    apply_rotary_pos_emb
    """
    sin, cos = (duplicate_interleave(t)[None, offset: tensor.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


class CodeGenAttention(nn.Cell):
    """
    CodeGenAttention
    """

    def __init__(self, config):
        super().__init__()

        max_positions = config.n_positions
        self.causal_mask = Parameter(ops.tril(ops.ones((max_positions, max_positions), dtype=mindspore.uint8)).view(
            1, 1, max_positions, max_positions
        ), requires_grad=False)

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        self.embed_dim = config.n_embd
        self.num_attention_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = ops.sqrt(mindspore.Tensor(self.head_dim, dtype=mindspore.float32)).astype(mindspore.float32)
        self.qkv_proj = nn.Dense(self.embed_dim, self.embed_dim * 3, has_bias=False)

        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, tensor, n_head, dim_head, mp_num):
        reshaped = tensor.reshape(tensor.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(tensor.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.transpose(0, 1, 3, 2, 4)
        elif len(tensor.shape) == 4:
            tensor = tensor.transpose(0, 2, 1, 3)
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

        # compute causal mask from causal mask buffer
        # query_length, key_length = query.shape(-2), key.shape(-2)
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.causal_mask[:, :, key_length - query_length: key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.astype(mindspore.float32)
        key = key.astype(mindspore.float32)

        attn_weights = ops.matmul(query, key.swapaxes(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        # mask_value = torch.finfo(attn_weights.dtype).min
        dtype_info = np.finfo(np.float32 if attn_weights.dtype == mindspore.float32 else np.float64)
        mask_value = Tensor(np.array([dtype_info.min]), dtype=attn_weights.dtype)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        # mask_value = Tensor(mask_value, dtype=attn_weights.dtype).astype(attn_weights.device)
        mask_value = Tensor(mask_value, dtype=attn_weights.dtype)
        attn_weights = mindspore.numpy.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(axis=-1)(attn_weights)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)

        return attn_output, attn_weights

    def construct(
            self,
            hidden_states: Optional[mindspore.Tensor],
            attention_mask: Optional[mindspore.Tensor] = None,
            layer_past: Optional[Tuple[mindspore.Tensor]] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Any, Optional[Tuple[Any, Any]]]:

        qkv = self.qkv_proj(hidden_states)

        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = mindspore.ops.split(qkv_split, local_dim, axis=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.transpose(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = mindspore.ops.cat([k_rot, k_pass], axis=-1)
            query = mindspore.ops.cat([q_rot, q_pass], axis=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.transpose(0, 2, 1, 3)
        query = query.transpose(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = mindspore.ops.cat((past_key, key), axis=-2)
            value = mindspore.ops.cat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
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
    CodeGenMLP
    """

    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Dense(embed_dim, intermediate_size)
        self.fc_out = nn.Dense(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def construct(self, hidden_states: Optional[mindspore.Tensor]) -> mindspore.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.gptj.modeling_gptj.GPTJBlock with GPTJ->CodeGen
class CodeGenBlock(nn.Cell):
    """CodeGenBlock"""
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm((config.n_embd,), epsilon=config.layer_norm_epsilon)
        self.attn = CodeGenAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)

    def construct(
            self,
            hidden_states: Optional[mindspore.Tensor],
            layer_past: Optional[Tuple[mindspore.Tensor]] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
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

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)
