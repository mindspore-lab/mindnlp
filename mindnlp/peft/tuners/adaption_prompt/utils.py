# Copyright 2023-present the HuggingFace Inc. team.
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
"""Utility functions for adaption prompt tuners."""
import inspect
from mindspore import Tensor
from mindnlp.core import nn, ops


def llama_rotate_half(x: Tensor) -> Tensor:
    """
    Rotate half the hidden dims of the input.

    This function was duplicated verbatim from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L126

    This was done to eliminate the Llama transformers implementation as a dependency of this file. Note that some other
    functions were also adapted from the transformers implementation but were modified.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), -1)


def llama_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """
    Apply rotary position embedding to query states in the Llama model using MindSpore.
    """
    if cos.ndim == 4:
        gather_indices = ops.broadcast_to(position_ids[:, None, :, None], (1, cos.shape[1], 1, cos.shape[3]))
        cos = ops.gather_elements(cos, 2, gather_indices)
        sin = ops.gather_elements(sin, 2, gather_indices)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (llama_rotate_half(q) * sin)
    return q_embed


def llama_compute_query_states(model: nn.Module, **kwargs) -> Tensor:
    """
    Computes query states for a neural network model.
    
    Args:
        model (nn.Module): The neural network model for which query states are computed.
    
    Returns:
        Tensor: The computed query states.
    
    Raises:
        ValueError: If the input parameters do not meet the required constraints.
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.shape
    query_states = ops.permute(model.q_proj(hidden_states).reshape(bsz, q_len, model.num_heads, model.head_dim), (0, 2, 1, 3))

    factor = model.k_proj.in_channels // model.k_proj.out_channels
    value_states = ops.permute(model.v_proj(hidden_states).reshape(bsz, q_len, (model.num_heads // factor), model.head_dim), (0, 2, 1, 3))

    seq_len = q_len
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            seq_len += past_key_value[0].shape[-2]
        else:
            seq_len += past_key_value.get_seq_length(model.layer_idx)

    if "position_ids" not in inspect.signature(model.rotary_emb).parameters:
        cos, sin = model.rotary_emb(value_states, seq_len=seq_len)
        return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)

    past_seen_tokens = 0
    if position_ids is None:
        if past_key_value is None:
            new_cache_positions = Tensor(ops.arange(q_len, q_len + q_len))
        else:
            past_seen_tokens = past_key_value.get_usable_length(q_len, model.layer_idx)
            new_cache_positions = Tensor(ops.arange(past_seen_tokens, past_seen_tokens + q_len))
        position_ids = ops.unsqueeze(new_cache_positions, 0)

    rotary_emb_kwargs = {"position_ids": position_ids}
    if "seq_len" in inspect.signature(model.rotary_emb).parameters:
        rotary_emb_kwargs["seq_len"] = seq_len

    cos, sin = model.rotary_emb(value_states, **rotary_emb_kwargs)
    if cos.shape[0] == 3:
        cos = ops.unsqueeze(cos, 1)
        sin = ops.unsqueeze(sin, 1)

    return (query_states * cos) + (llama_rotate_half(query_states) * sin)


def is_adaption_prompt_trainable(params: str) -> bool:
    """Return True if cell is trainable under adaption prompt fine-tuning."""
    return params.split(".")[-1].startswith("adaption_")
