# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import logging
from typing import Dict, List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error

from ..tokenizer.normalization import NormalizedToken, unnormalize_token

LOG = logging.getLogger(__name__)


def well_trained_tokens(
    vocab: Dict[NormalizedToken, int],
    embed: Tensor,
    lm_head: Optional[Tensor],
    known_unused: Optional[List[NormalizedToken]] = None,
    quantile: float = 0.01,
) -> List[NormalizedToken]:
    """Get a list of tokens that are well-trained in the model.

    Uses the approach from "Fishing for Magikarp: Automatically Detecting
    Under-trained Tokens in Large Language Models"
    (https://arxiv.org/abs/2405.05417).

    Args:
        vocab: The vocabulary of the model, mapping tokens to indices.
        embed: The input embedding matrix of the model.
        lm_head: The output embedding matrix of the model (optional).
        known_unused: A list of known unused tokens (optional).
        quantile: The quantile to use for filtering (default: 0.01).

    Returns:
        A list of tokens that can be assumed to be well-trained in the model.
    """
    unused_indices = set(range(embed.shape[0])) - set(vocab.values())
    if known_unused:
        unused_indices.update(vocab[tok] for tok in known_unused if tok in vocab)
    for tok in vocab:
        tok_text = unnormalize_token(tok)
        if "unused_token" in tok_text or "reserved_special_token" in tok_text:
            LOG.debug(f"Assuming {tok_text} is unused")
            unused_indices.add(vocab[tok])

    if unused_indices:
        mean_unused_in = embed[list(unused_indices)].mean(axis=0)
        mean_unused_out = (
            lm_head[list(unused_indices)].mean(axis=0) if lm_head is not None else None
        )
        LOG.info(f"Found {len(unused_indices)} unused tokens")
    else:
        mean_unused_in = None
        mean_unused_out = None

    bad_indices = set(unused_indices)

    if lm_head is not None:
        l2_norms = ops.norm(embed, axis=1).astype(mindspore.float32)  # pylint: disable=unexpected-keyword-arg
        threshold = _quantile(l2_norms, quantile)
        LOG.debug(
            f"Unused token L2 norm threshold: {threshold.asnumpy():.4f} "
            f"({int(quantile * 100)}th percentile)"
        )
        l2_bad_indices = ops.nonzero(l2_norms < threshold).squeeze(1)
        if l2_bad_indices.shape[0] > 0:
            bad_indices.update(l2_bad_indices.asnumpy().tolist())
            LOG.info(f"Discarding {l2_bad_indices.shape[0]} low-l2 tokens")

    if mean_unused_in is not None:
        cos_sim = ops.cosine_similarity(  # pylint: disable=unexpected-keyword-arg
            embed.astype(mindspore.float32),
            mean_unused_in.unsqueeze(0).astype(mindspore.float32),
            axis=1,
        )
        threshold = _quantile(cos_sim, 1 - quantile)
        LOG.debug(
            f"Unused token threshold in embed_tokens: {threshold.asnumpy():.4f} "
            f"({int((1 - quantile) * 100)}th percentile)"
        )
        if threshold < 0.5:
            threshold = Tensor(0.5, dtype=mindspore.float32)
            LOG.debug("Clamping threshold to 0.5")
        cos_bad_indices = ops.nonzero(cos_sim > threshold).squeeze(1)
        if cos_bad_indices.shape[0] > 0:
            bad_indices.update(cos_bad_indices.asnumpy().tolist())
            LOG.info(
                f"Discarding {cos_bad_indices.shape[0]} high-sim to unused mean tokens"
            )

    if lm_head is not None and mean_unused_out is not None:
        cos_sim = ops.cosine_similarity(  # pylint: disable=unexpected-keyword-arg
            lm_head.astype(mindspore.float32),
            mean_unused_out.unsqueeze(0).astype(mindspore.float32),
            axis=1,
        )
        threshold = _quantile(cos_sim, 1 - quantile)
        LOG.debug(
            f"Unused token threshold in lm_head: {threshold.asnumpy():.4f} "
            f"({int((1 - quantile) * 100)}th percentile)"
        )
        if threshold < 0.5:
            threshold = Tensor(0.5, dtype=mindspore.float32)
            LOG.debug("Clamping threshold to 0.5")
        cos_bad_indices = ops.nonzero(cos_sim > threshold).squeeze(1)
        if cos_bad_indices.shape[0] > 0:
            bad_indices.update(cos_bad_indices.asnumpy().tolist())
            LOG.info(
                f"Discarding {cos_bad_indices.shape[0]} high-sim to unused mean tokens"
            )

    good_tokens = [tok for tok, idx in vocab.items() if idx not in bad_indices]
    LOG.info(
        f"Found {len(good_tokens)} well-trained tokens, {len(bad_indices)} bad tokens"
    )
    return good_tokens


def _quantile(tensor: Tensor, q: float) -> Tensor:
    """Compute quantile of a 1-D tensor (MindSpore-compatible)."""
    sorted_tensor = ops.sort(tensor.flatten())[0]
    n = sorted_tensor.shape[0]
    idx = int(q * (n - 1))
    idx = max(0, min(idx, n - 1))
    return sorted_tensor[idx]
