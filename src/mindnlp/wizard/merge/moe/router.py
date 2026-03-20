# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#
# Key changes:
#   - torch.no_grad() → mindspore.ops.stop_gradient / removed (PYNATIVE_MODE has no grad by default)
#   - torch.linalg.cond() → manual SVD-based implementation
#   - torch.nn.functional.one_hot() → ops.one_hot()
#   - torch.randn/rand → ops.randn/rand
#   - AutoModelForCausalLM → try mindnlp.transformers first

import logging
import math
from typing import Dict, List

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
import tqdm

try:
    from mindnlp.transformers import (
        AutoModelForCausalLM,
        PreTrainedTokenizerBase,
        BatchEncoding,
    )
except ImportError:
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, BatchEncoding

from ..common import ModelReference
from .config import Expert


def get_hidden_states(
    model,
    tokenized: BatchEncoding,
    average: bool = True,
) -> List[mindspore.Tensor]:
    output = model(
        **dict(tokenized.items()),
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = ops.stack(
        output.hidden_states[:-1]
    )  # (num_layers, batch_size, seq_len, hidden_size)
    if average:
        hidden_states = hidden_states.sum(axis=2) / hidden_states.shape[2]
    else:
        hidden_states = hidden_states[:, :, -1, :]
    return hidden_states.sum(axis=1) / hidden_states.shape[1]


def get_cheap_embedding(
    embed: mindspore.Tensor,
    tokenized: Dict[str, mindspore.Tensor],
    num_layers: int,
    vocab_size: int,
) -> mindspore.Tensor:
    input_ids = tokenized["input_ids"]
    onehot = ops.one_hot(
        input_ids, vocab_size, mindspore.Tensor(1.0), mindspore.Tensor(0.0)
    )  # (batch_size, seq_len, vocab_size)
    h = onehot.float() @ embed.float()  # (batch_size, seq_len, hidden_size)
    embedded = (
        (h * tokenized["attention_mask"].unsqueeze(-1))
        .sum(axis=1)
        .sum(axis=0, keepdims=True)
    )  # (1, hidden_size)
    norm_val = ops.norm(embedded, dim=-1, keepdim=True).clamp(min=1e-8)
    res = embedded / norm_val  # (1, hidden_size)
    return res.repeat(num_layers, 1)


def tokenize_prompts(
    prompts: List[str], tokenizer: PreTrainedTokenizerBase
):
    return tokenizer(
        [(tokenizer.bos_token or "") + p for p in prompts],
        return_tensors="ms",
        padding=True,
        add_special_tokens=False,
    )


def get_gate_params(  # pylint: disable=too-many-positional-arguments
    model_ref: ModelReference,
    tokenizer: PreTrainedTokenizerBase,
    experts: List[Expert],
    mode: str = "hidden",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lazy_loader: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
):
    gate_vecs = []
    _do_it = None

    model_cfg = model_ref.config(trust_remote_code=trust_remote_code)

    if mode == "random":
        return ops.randn(
            (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
        )
    elif mode == "uniform_random":
        in_features = model_cfg.hidden_size
        scale = math.sqrt(1.0 / in_features)
        return (
            ops.rand(
                (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
            )
            * 2
            * scale
            - scale
        )
    elif mode == "cheap_embed":
        embed = model_ref.lazy_loader(lazy_loader=lazy_loader).get_tensor(
            "model.embed_tokens.weight"
        )

        def _do_it(tokenized):  # pylint: disable=function-redefined
            return get_cheap_embedding(
                embed,
                tokenized,
                num_layers=model_cfg.num_hidden_layers,
                vocab_size=model_cfg.vocab_size,
            )

    elif mode in ("hidden", "hidden_avg", "hidden_last"):
        model = AutoModelForCausalLM.from_pretrained(
            model_ref.model.path,
            revision=model_ref.model.revision,
            ms_dtype=mindspore.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
        )

        def _do_it(tokenized):  # pylint: disable=function-redefined
            return get_hidden_states(
                model, tokenized=tokenized, average=mode == "hidden_avg"
            )

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
        hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))
        if expert.negative_prompts:
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )

        norm_val = ops.norm(hidden_states, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
        hidden_states = hidden_states / norm_val
        gate_vecs.append(hidden_states)
    gate_vecs = ops.stack(gate_vecs, axis=0)  # (num_expert, num_layer, hidden_size)
    return gate_vecs.permute(1, 0, 2)


def _cond_via_svd(matrix: mindspore.Tensor) -> mindspore.Tensor:
    """Compute the condition number of a matrix via SVD (replaces torch.linalg.cond)."""
    _, svd_values, _ = ops.svd(matrix)
    return svd_values.max() / svd_values.min()


def warn_degenerate_gates(gate_vecs: mindspore.Tensor, threshold: float = 5.0):
    degen_indices = []
    num_layers, _num_experts, _hidden_size = gate_vecs.shape
    for idx in range(num_layers):
        c = _cond_via_svd(gate_vecs[idx, :, :].float())
        if c > threshold:
            degen_indices.append(idx)

    if degen_indices:
        if len(degen_indices) == 1:
            layer_str = f"layer {degen_indices[0]}"
            verb = "has"
        elif len(degen_indices) == 2:
            layer_str = f"layers {' and '.join(map(str, degen_indices))}"
            verb = "have"
        elif len(degen_indices) >= num_layers:
            layer_str = "ALL layers"
            verb = "have"
        else:
            layer_str = (
                "layers "
                + ", ".join(map(str, degen_indices[:-1]))
                + ", and "
                + str(degen_indices[-1])
            )
            verb = "have"

        logging.warning(
            f"{layer_str} {verb} degenerate routing parameters "
            "- your prompts may be too similar."
        )
        logging.warning("One or more experts will be underutilized in your model.")
