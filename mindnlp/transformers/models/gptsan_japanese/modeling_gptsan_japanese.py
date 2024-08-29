# coding=utf-8
# Copyright 2023 Toshiyuki Sakamoto(tanreinama) and HuggingFace Inc. team.
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
""" PyTorch GPTSANJapanese model."""


import copy
from typing import List, Optional, Tuple, Union

import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.ops import OneHot
from mindspore.common.initializer import initializer, Constant, Normal
from mindnlp.core import nn, ops
import mindnlp.core.nn.functional as F

from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ....utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    logging,
)
from .configuration_gptsan_japanese import GPTSanJapaneseConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPTSanJapaneseConfig"
_CHECKPOINT_FOR_DOC = "Tanrei/GPTSAN-japanese"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.router_z_loss_func
def router_z_loss_func(router_logits: mindspore.Tensor) -> float:
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = ops.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return ops.sum(z_loss) / (num_groups * tokens_per_group)


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.load_balancing_loss_func
def load_balancing_loss_func(router_probs: mindspore.Tensor, expert_indices: mindspore.Tensor) -> float:
    num_experts = router_probs.shape[-1]
    if expert_indices.dtype != mindspore.int64:
        expert_indices = expert_indices.astype(mindspore.int64)
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    expert_mask = F.one_hot(expert_indices, num_experts)
    expert_mask = ops.max(expert_mask, dim=-2).values
    expert_mask = expert_mask.to(mindspore.float32)
    tokens_per_group_and_expert = ops.mean(expert_mask, dim=-2)
    router_prob_per_group_and_expert = ops.mean(router_probs, dim=-2)
    return ops.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class GPTSanJapaneseDenseActDense(nn.Module):
    """
    FFN Layer for Switch Transformer and Extra layers

    GPTSAN can mix Switch Transformer layers and normal Transformer layers This class is used as Expert in Switch
    Transformer layers and as FFN in regular Transformer layers. RELU is used in the Switch Transformer layer, and
    Swish is used in the normal Transformer layer, so there is a choice of which is used in the argument.

    """

    def __init__(self, config: GPTSanJapaneseConfig, ext_layer=False):
        super().__init__()
        d_inter = config.d_ext if ext_layer else config.d_ff
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)
        self.act = ACT2FN["swish" if ext_layer else "relu"]

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`mindspore.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            mindspore.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router with SwitchTransformers->GPTSanJapanese
class GPTSanJapaneseTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(mindspore, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`mindspore.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`mindspore.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`mindspore.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)
        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= ops.uniform(hidden_states.shape,
                                         mindspore.Tensor(1.0 - self.jitter_noise, hidden_states.dtype),
                                         mindspore.Tensor(1.0 + self.jitter_noise, hidden_states.dtype),
                                         seed=0,
                                         dtype=hidden_states.dtype)
        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)
        router_logits = router_logits.astype(hidden_states.dtype)
        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: mindspore.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)
        expert_index = ops.argmax(router_probs, dim=-1)
        one_hot_op = OneHot()
        expert_index = one_hot_op(expert_index, self.num_experts, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
        token_priority = ops.cumsum(expert_index, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
        router_probs, _ = ops.max(router_probs, dim=-1)
        return expert_index, router_probs.astype(mindspore.float32), router_logits.astype(mindspore.float32)

# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP with SwitchTransformers->GPTSanJapanese
class GPTSanJapaneseSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """
    def __init__(self, config: GPTSanJapaneseConfig, expert_class: nn.Module = GPTSanJapaneseDenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = GPTSanJapaneseTop1Router(config)

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states):
        router_mask, router_probs, router_logits = self.router(hidden_states)

        expert_index = ops.argmax(router_mask, dim=-1)
        next_states = hidden_states.copy()

        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
        router_probs = router_probs.unsqueeze(-1)
        forwarded_states = router_probs * next_states

        if forwarded_states.shape != hidden_states.shape:
            forwarded_states = ops.reshape(forwarded_states, hidden_states.shape)
        return forwarded_states, (router_logits, expert_index)

class GPTSanJapaneseLayerSparseFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.
    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        self.mlp = GPTSanJapaneseSparseMLP(config)
        self.soft_bypass_mlp = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, output_router_tuple):
        forwarded_states, router_tuple = self.mlp(hidden_states)
        if forwarded_states.shape != hidden_states.shape:
            forwarded_states = ops.reshape(forwarded_states, hidden_states.shape)
        forwarded_states += ops.tanh(self.soft_bypass_mlp(hidden_states))
        output = hidden_states + self.norm(forwarded_states)
        if output_router_tuple and router_tuple is not None:
            return output, router_tuple
        else:
            return output

class GPTSanJapaneseLayerDenseFF(nn.Module):
    r"""
    Extra Transformers Feed Forward layer module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        # Check if it is a sparse layer, if not then it is a dense layer
        self.mlp = GPTSanJapaneseDenseActDense(config, ext_layer=True)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`mindspore.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            mindspore.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        forwarded_states = self.mlp(hidden_states)
        output = hidden_states + self.norm(forwarded_states)
        return output

# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->GPTSanJapanese
class GPTSanJapaneseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[GPTSanJapaneseConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Convert weights to Float32
        self.k_proj.weight.set_data(self.k_proj.weight.astype(mindspore.float32))
        self.v_proj.weight.set_data(self.v_proj.weight.astype(mindspore.float32))
        self.q_proj.weight.set_data(self.q_proj.weight.astype(mindspore.float32))
        self.out_proj.weight.set_data(self.out_proj.weight.astype(mindspore.float32))

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

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

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        hidden_states = hidden_states.astype(mindspore.float32)
        # Get query projection
        query_states = self.q_proj(hidden_states).astype(mindspore.float32) * self.scaling

        # Get key, value projections
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # Reuse key, value from past_key_value for cross-attention
            key_states = past_key_value[0].astype(mindspore.float32)
            value_states = past_key_value[1].astype(mindspore.float32)
        elif is_cross_attention:
            # Cross-attention: calculate new key and value
            key_states = self._shape(self.k_proj(key_value_states).astype(mindspore.float32), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states).astype(mindspore.float32), -1, bsz)
        elif past_key_value is not None:
            # Self-attention: reuse key, value from past_key_value
            key_states = self._shape(self.k_proj(hidden_states).astype(mindspore.float32), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states).astype(mindspore.float32), -1, bsz)
            key_states = ops.cat([past_key_value[0].astype(mindspore.float32), key_states], dim=2)
            value_states = ops.cat([past_key_value[1].astype(mindspore.float32), value_states], dim=2)
        else:
            # Self-attention: calculate new key and value
            key_states = self._shape(self.k_proj(hidden_states).astype(mindspore.float32), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states).astype(mindspore.float32), -1, bsz)

        if self.is_decoder:
            # Save past key and value states for future use
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape).astype(mindspore.float32)
        key_states = key_states.reshape(*proj_shape).astype(mindspore.float32)
        value_states = value_states.reshape(*proj_shape).astype(mindspore.float32)

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
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).astype(mindspore.float32) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).astype(mindspore.float32)

        attn_weights = ops.softmax(attn_weights, dim=-1).astype(mindspore.float32)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1).astype(mindspore.float32) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len).astype(mindspore.float32)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len).astype(mindspore.float32)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).astype(mindspore.float32)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len).astype(mindspore.float32)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs.astype(mindspore.float32), value_states.astype(mindspore.float32))

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).astype(mindspore.float32)
        attn_output = attn_output.swapaxes(1, 2).astype(mindspore.float32)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim).astype(mindspore.float32)
        attn_output = self.out_proj(attn_output.astype(self.out_proj.weight.dtype)).astype(mindspore.float32)
        return attn_output, attn_weights_reshaped, past_key_value


class GPTSanJapaneseLayerSelfAttention(nn.Module):
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.self_attn = GPTSanJapaneseAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            is_decoder=True,
            bias=has_relative_attention_bias,
        )
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: Optional[Tuple[mindspore.Tensor]],
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:
        r"""
        Self-attention and normalize block.

        Args:
            hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding. If `past_key_values` are used, the user can optionally input only the last
                `decoder_input_ids` (those that don't have their past key value states given to this model) of shape
                `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            head_mask (`numpy.ndarray` of shape `({0})`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            Tuple[mindspore.Tensor[num_groups, tokens_per_group, hidden_dim],...]
        """
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        atten_out = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask = attention_mask * float(ops.finfo(hidden_states.dtype).min),
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
            )
        if output_attentions:
            attn_weights = (atten_out[1],)
        else:
            attn_weights = ()

        attention_output = atten_out[0]

        hidden = hidden_states + self.norm(attention_output)

        if use_cache:
            outputs = (hidden, atten_out[2])  # hidden, present, (attentions)
        else:
            outputs = (hidden,)  # hidden, (attentions)

        return outputs + attn_weights


class GPTSanJapaneseBlock(nn.Module):
    def __init__(self, config, ext_layer=False):
        super().__init__()
        self.self_attn = GPTSanJapaneseLayerSelfAttention(config)
        self.feed_forward = GPTSanJapaneseLayerDenseFF(config) if ext_layer else GPTSanJapaneseLayerSparseFF(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[mindspore.Tensor]],
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_router_tuple: Optional[bool] = False,
    ) -> Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:

        atten_out = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attention_output = atten_out[0]

        if isinstance(self.feed_forward, GPTSanJapaneseLayerSparseFF):
            sparse_out = self.feed_forward(attention_output, output_router_tuple)
            if output_router_tuple:
                hidden, router_tuple = sparse_out
            else:
                hidden = sparse_out
        else:
            hidden = self.feed_forward(attention_output)

        outputs = (hidden,) + atten_out[1:]

        if isinstance(self.feed_forward, GPTSanJapaneseLayerSparseFF) and output_router_tuple:
            outputs += (router_tuple,)

        return outputs



class GPTSanJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTSanJapaneseConfig
    base_model_prefix = "gptsan_japanese"
    supports_gradient_checkpointing = False
    _no_split_modules = ["GPTSanJapaneseBlock"]
    _skip_keys_device_placement = "past_key_values"

    @property
    def dummy_inputs(self):
        input_ids = mindspore.Tensor(DUMMY_INPUTS)
        input_mask = mindspore.Tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, nn.LayerNorm):
            module.weight.set_data(initializer(Constant(factor * 1.0), module.weight.shape, module.weight.dtype))
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSanJapaneseModel):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.embed_tokens.weight.set_data(
                initializer(Normal(0.0,factor * 1.0),
                            module.embed_tokens.weight.shape, module.embed_tokens.weight.dtype))
            module.position_embeddings.weight.set_data(
                initializer(Normal(0.0,factor * 1.0),
                            module.position_embeddings.weight.shape, module.position_embeddings.weight.dtype))
            if hasattr(module, "extra_position_embeddings") and module.extra_position_embeddings is not None:
                module.extra_position_embeddings.weight.data.set_data(initializer(Normal(0.0,factor * 1.0), \
                                                    module.extra_position_embeddings.weight.data.shape, \
                                                    module.extra_position_embeddings.weight.data.dtype))
        elif isinstance(module, (GPTSanJapaneseModel, GPTSanJapaneseForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            nn.init.normal_(module.final_logits_bias,mean=0.0, std=factor * 1.0)

            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.set_data(initializer(Normal(0.0,factor * 1.0), \
                                                    module.lm_head.weight.data.shape, \
                                                    module.lm_head.weight.data.dtype))
        elif isinstance(module, GPTSanJapaneseDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.set_data(initializer(Normal(0.0,factor * ((self.config.d_model) ** -0.5)),
                                                module.wi.weight.shape, module.wi.weight.dtype))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.set_data(initializer('zeros', module.wi.bias.shape, module.wi.bias.dtype))
            module.wo.weight.set_data(initializer(Normal(0.0,factor * ((self.config.d_ff) ** -0.5)),
                                                module.wo.weight.shape, module.wo.weight.dtype))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.set_data(initializer('zeros', module.wo.bias.shape, module.wo.bias.dtype))
        elif isinstance(module, GPTSanJapaneseAttention):
            # Multi-headed attention
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.k_proj.weight.set_data(initializer(Normal(0.0,factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                               module.k_proj.weight.shape, module.k_proj.weight.dtype))
            module.v_proj.weight.set_data(initializer(Normal(0.0,factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                               module.v_proj.weight.shape, module.v_proj.weight.dtype))
            module.q_proj.weight.set_data(initializer(Normal(0.0,factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                               module.q_proj.weight.shape, module.q_proj.weight.dtype))
            module.out_proj.weight.set_data(initializer(Normal(0.0,factor * ((n_heads * key_value_proj_dim) ** -0.5)),
                                               module.out_proj.weight.shape, module.out_proj.weight.dtype))
        elif isinstance(module, GPTSanJapaneseSparseMLP):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.router.classifier.weight.data.set_data(initializer(Normal(0.0,factor * 1.0), \
                                                              module.router.classifier.weight.data.shape, \
                                                              module.router.classifier.weight.data.dtype))
            for idx in range(self.config.num_experts):
                module.experts[f"expert_{idx}"].wi.weight.set_data(initializer(Normal(0.0,factor * (d_model**-0.5)), \
                                                              module.experts[f"expert_{idx}"].wi.weight.data.shape, \
                                                              module.experts[f"expert_{idx}"].wi.weight.data.dtype))
                module.experts[f"expert_{idx}"].wo.weight.set_data(initializer(Normal(0.0,factor * (d_model**-0.5)), \
                                                              module.experts[f"expert_{idx}"].wo.weight.data.shape, \
                                                              module.experts[f"expert_{idx}"].wo.weight.data.dtype))
       # Ensure `last_project` bias is initialized to zero
        if hasattr(module, 'last_project') and hasattr(module.last_project, 'bias') and module.last_project.bias is not None:
            module.last_project.bias.set_data(initializer('zeros', module.last_project.bias.shape, module.last_project.bias.dtype))
    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


GPTSAN_JAPANESE_START_DOCSTRING = r"""

    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPTSAN_JAPANESE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. GPTSAN-japanese is a model that generates sentence
            continuations or predicts tokens at mask positions. Special tokens required for inputs to the model are
            automatically appended.
        attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            An input that masks the Prefix part in the Prefix-LM input. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **prefix** input,
            - 0 for tokens that are **not-prefix** input.
        spout (`mindspore.Tensor` of shape `(batch_size, config.d_spout)`):
                This vector is transformed through an 8-layer FFN and can be used instead of `past_key_values`.
        past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        head_mask (`mindspore.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`mindspore.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        router_logits (`tuple(mindspore.Tensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
            Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.
"""


class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.config = copy.deepcopy(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        self.act = ACT2FN["swish"]

        self.blocks = nn.ModuleList([])
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSanJapaneseBlock(config))
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSanJapaneseBlock(config, ext_layer=True))

        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        spout: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        num_precontext: Optional[mindspore.Tensor] = None,
    ) -> Union[MoEModelOutputWithPastAndCrossAttentions, Tuple[mindspore.Tensor]]:
        r"""
        num_precontext (`mindspore.Tensor` of shape `(batch_size,1)`):
            length of `hybrid` input tokens in the input. Tokens up to this length refer to both front and back like
            BERT, tokens after that refer only to front like GPT. see also:
            https://github.com/tanreinama/GPTSAN/blob/main/report/model.md

        Returns:
            `MoEModelOutputWithPastAndCrossAttentions` or `tuple` if `return_dict` returns
            MoEModelOutputWithPastAndCrossAttentions insted of tuple
        """

        if input_ids is not None:
            input_ids = input_ids.astype(mindspore.int32)
        if attention_mask is not None:
            attention_mask = attention_mask.astype(mindspore.int32)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.astype(mindspore.int32)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            input_ids = ops.zeros([1, 1]).int() # dummy for input_ids was None
        if inputs_embeds is not None:
            raise NotImplementedError(
                "GPTSanJapaneseModel does not use `inputs_embeds`. Make sure to pass in `input_ids` instead."
            )
        num_pasts_contexts = 0
        num_batch = input_ids.shape[0]
        pasts_or_spout_value = None
        if past_key_values is not None:
            num_pasts_contexts = past_key_values[0][0].shape[2]
        elif self.config.d_spout and spout is not None:
            # `spout` is a special input vector specific to GPTSAN
            # This controls the output by projecting embedded information such as the class of sentences during learning.
            # It should passed instead of the first past_key_value.
            # See the original GPTSAN repository for details
            num_pasts_contexts += 1

        # If there is an attention_mask, increase first one for spout
        if self.config.d_spout and spout is not None and attention_mask is not None:
            attention_mask_with_spout = ops.ones(num_batch, attention_mask.shape[1] + 1)
            attention_mask_with_spout[:, 1:] -= 1 - attention_mask  # 1st token should be spout
            attention_mask = attention_mask_with_spout  # update attention_mask

        if num_precontext is not None:
            # `num_precontext` is the number of tokens that refer to each other in prefix-lm
            # created per batch, so dimension of num_precontext should be [batch, 1]
            if not (
                len(num_precontext.shape) == 2 and num_precontext.shape[1] == 1
            ):  # num_precontext Should be [batch,1]
                raise ValueError("num_precontext should be [batch, 1] size.")
            num_precontext = ops.reshape(num_precontext, [-1])
        else:
            num_precontext = ops.zeros(num_batch).int()

        num_input_contexts = input_ids.shape[1]
        num_output_contexts = num_input_contexts + num_pasts_contexts

        hidden_states = self.embed_tokens(input_ids)

        if past_key_values is not None:
            pasts_or_spout_value = past_key_values
        elif self.config.d_spout and spout is not None:
            spout_output = self.spout(spout)
            expected_shape = [
                num_batch,
                self.config.num_layers,
                2,
                self.config.num_heads,
                num_pasts_contexts,
                self.config.d_model // self.config.num_heads,
            ]
            if spout_output.size != np.prod(expected_shape):
                raise ValueError(f"Cannot reshape spout_output of shape {spout_output.shape} to {expected_shape}")

            pasts_or_spout_value = ops.reshape(spout_output, expected_shape)
            if len(pasts_or_spout_value.shape) > 1:
                pasts_or_spout_value = ops.split(pasts_or_spout_value, [1] * self.config.num_layers, 1)
            else:
                raise ValueError(f"Unexpected shape for pasts_or_spout_value: {pasts_or_spout_value.shape}")


            # make same shape as past_key_values
            pasts_or_spout_value = tuple(([b.squeeze(1) for b in ops.split(a.squeeze(1), [1, 1], 1)]) for a in pasts_or_spout_value)

        else:
            pasts_or_spout_value = [None] * self.config.num_layers
        token_position = ops.arange(num_input_contexts) + num_pasts_contexts

        if attention_mask is None:
            attention_mask = ops.ones(num_batch, num_input_contexts)

        # positions for get position_embeddings
        gather_position = (
            (ops.zeros((num_batch, self.config.d_model, num_input_contexts)) + token_position.unsqueeze(0))
            .transpose(0, 2, 1)
            .long()
        )
        # When padding with padding_side="left", zeros line up on the left side of attention_mask, so position_embeddings is shifted accordingly
        gather_position -= (1 - attention_mask).argmin(axis=-1).unsqueeze(1).unsqueeze(2)
        gather_position = ops.clamp(gather_position, num_pasts_contexts, self.config.max_position_embeddings - 1)

        # attention_mask is applied per batch
        for i in range(num_batch):
            hidden_states[i] += ops.gather(self.position_embeddings.weight, dim=0, index=gather_position[i])
        # Create a mask to be used when making the prefix Input length of Prefix-LM variable
        causal_mask = (
            ops.tril(ops.ones((num_output_contexts, num_output_contexts), dtype=mindspore.uint8))
            .view(1, 1, num_output_contexts, num_output_contexts))
        prefix_lm_mask = causal_mask[:, :, -num_input_contexts:, :]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).unsqueeze(2).astype(mindspore.int32)
            prefix_lm_mask = ((prefix_lm_mask + token_type_ids) > 0).float()

        # Marge prefix_lm_mask and attention_mask
        extended_attention_mask = prefix_lm_mask * attention_mask.unsqueeze(1).unsqueeze(2)

        # Prepare head mask if needed
        if head_mask is not None:
            head_mask = self.get_head_mask(
                head_mask, self.config.num_switch_layers + self.config.num_ext_layers
            )  # n_layer x batch x n_heads x N x N

        # outputs
        present_key_value_states = () if self.config.use_cache or use_cache else None
        all_hidden_states = () if self.config.output_hidden_states or output_hidden_states else None
        all_attentions = () if self.config.output_attentions or output_attentions else None
        all_router_probs = () if self.config.output_router_logits or output_router_logits else None

        for layer, past in enumerate(pasts_or_spout_value):
            if layer == self.config.num_switch_layers:
                if self.config.num_ext_layers > 0:
                    # extra_position_embeddings are extra position embeddings that are only created when extending the model with code from the original GPTSAN repository. Not used in the default model.
                    # However, it is created when you create an additional layer and partially train only that location.
                    # Therefore, convert_gptsan_tf_checkpoint_to_pytorch.py is used when converting and loading models created in the original GPTSAN repository.
                    for i in range(num_batch):
                        hidden_states[i] += ops.gather(
                            self.extra_position_embeddings.weight, dim=0, index=gather_position[i]
                        )

            output_router_tuple = (
                self.config.output_router_logits or output_router_logits
            ) and layer < self.config.num_switch_layers
            block_output = self.blocks[layer](
                hidden_states=hidden_states,
                past_key_value=past,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                use_cache=self.config.use_cache or use_cache,
                output_attentions=self.config.output_attentions or output_attentions,
                output_router_tuple=output_router_tuple,
            )

            outpos = 0
            hidden_states = block_output[outpos]
            if self.config.output_hidden_states or output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.config.use_cache or use_cache:
                outpos += 1
                present = block_output[outpos]
                present_key_value_states += (present,)
            if self.config.output_attentions or output_attentions:
                outpos += 1
                attention_probs = block_output[outpos]
                all_attentions += (attention_probs,)
            if output_router_tuple:
                outpos += 1
                router_tuple = block_output[outpos]
                all_router_probs.append(router_tuple[0])

        hidden_states = self.last_project(hidden_states)
        hidden_states = self.act(hidden_states)

        if self.config.output_hidden_states or output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_router_probs,
                ]
                if v is not None
            )

        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_probs=all_router_probs,
        )



class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPTSanJapaneseConfig, dtype=None):
        super().__init__(config)
        self.model = GPTSanJapaneseModel(config)
        self.register_buffer("final_logits_bias", ops.zeros(1, config.vocab_size))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight

        if dtype is not None:
            self.to(dtype=dtype)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        spout: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[MoECausalLMOutputWithPast, Tuple[mindspore.Tensor]]:\

        input_ids = input_ids.astype(mindspore.int32) if input_ids is not None else None
        attention_mask = attention_mask.astype(mindspore.int32) if attention_mask is not None else None
        token_type_ids = token_type_ids.astype(mindspore.int32) if token_type_ids is not None else None
        labels = labels.astype(mindspore.int32) if labels is not None else None


        SEG_TOKEN = self.config.separator_token_id
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_return_dict = True
        num_precontext = None
        if input_ids is not None:
            num_batch = input_ids.shape[0]
            num_precontext = ops.zeros(num_batch).int()
            where_separators = ops.nonzero(input_ids == SEG_TOKEN)
            if where_separators.size != 0:  # 检查 where_separators 是否为空
                for index in where_separators:
                    batch_idx = index[0].asnumpy().item()
                    if batch_idx < num_precontext.shape[0]:
                        num_precontext[batch_idx] += index[1]
            num_precontext = num_precontext.unsqueeze(1)

        # 调用模型的前向传播
        outputs = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            spout,
            past_key_values,
            head_mask,
            use_cache,
            inputs_embeds,
            decoder_inputs_embeds,
            output_attentions,
            output_hidden_states,
            model_return_dict,
            output_router_logits,
            num_precontext,
        )

        lm_logits = self.lm_head(outputs[0]).astype(mindspore.float32)

        lm_logits = self.lm_head(outputs[0])
        if lm_logits.shape[-1] == self.final_logits_bias.shape[-1]:
            lm_logits = lm_logits + self.final_logits_bias

        loss = None
        z_loss = None
        router_probs = None
        aux_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            if output_router_logits:
                router_logits, expert_indexes = self._unpack_router_logits(outputs.router_probs)
                z_loss = router_z_loss_func(router_logits)
                router_probs = nn.Softmax(dim=-1)(router_logits)
                aux_loss = load_balancing_loss_func(router_probs, expert_indexes)
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    lm_logits,
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.router_probs,
                    z_loss,
                    aux_loss,
                ]
                if v is not None
            )

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_probs,
            z_loss=z_loss,
            aux_loss=aux_loss,
        )





    def prepare_inputs_for_generation(
        self,
        input_ids: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        token_type_ids: Optional[mindspore.Tensor] = None,
        spout: Optional[Union[List, mindspore.Tensor]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        **kwargs,
    ):
        if isinstance(spout, list):
            spout = mindspore.Tensor(spout).float()
        if past_key_values is not None:
            return {
                "input_ids": input_ids[:, -1:] if input_ids is not None else None,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids[:, -1:] if token_type_ids is not None else None,
                "spout": spout,
                "past_key_values": past_key_values,
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "spout": spout,
            "past_key_values": None,
        }

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.prepare_decoder_input_ids_from_labels with SwitchTransformers->GPTSanJapanese
    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        return self._shift_right(labels)

    # Copied from transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration.resize_token_embeddings with MBart->GPTSanJapanese
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # Copied from transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration._resize_final_logits_bias with MBart->GPTSanJapanese
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.set_output_embeddings with SwitchTransformers->GPTSanJapanese
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.get_output_embeddings with SwitchTransformers->GPTSanJapanese
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration._unpack_router_logits with SwitchTransformers->GPTSanJapanese
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        return ops.cat(total_router_logits, dim=1), ops.cat(total_expert_indexes, dim=1)
__all__=["GPTSanJapaneseForConditionalGeneration",
        "GPTSanJapaneseModel",
        "GPTSanJapanesePreTrainedModel",]
