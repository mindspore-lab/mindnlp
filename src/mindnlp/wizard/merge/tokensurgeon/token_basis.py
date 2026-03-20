# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import logging
from typing import Dict, List, Tuple

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error

from ..tokenizer.normalization import NormalizedToken
from .omp import batch_omp

LOG = logging.getLogger(__name__)


def sparse_linear_basis(
    points: Tensor,
    k: int,
    d: int,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    Form an approximate orthogonal basis from sparse linear combinations of input points.
    Args:
        points: (num_pts, embed_dim) tensor of input points
        k: number of points to select per basis vector
        d: dimensionality of the basis
        eps: numerical stability parameter
    Returns:
        indices: (d, k) tensor of selected indices
        coeffs: (d, k) tensor of coefficients for each selected point
    """
    assert points.ndim == 2
    num_pts, embed_dim = points.shape
    assert k <= num_pts, "k must be less than or equal to the number of points"
    assert d <= embed_dim, "d must be less than or equal to the embedding dimension"

    mean_embed = points.mean(axis=0)
    centered_embeddings = (points - mean_embed).astype(mindspore.float32)
    covariance_matrix = (
        centered_embeddings.T @ centered_embeddings
    ) / num_pts  # (embed_dim, embed_dim)

    U, _S, _V = ops.svd(covariance_matrix)
    U_d = U[:, :d]  # (embed_dim, d)

    indices, coeffs = batch_omp(
        U_d.T,  # (d, embed_dim)
        centered_embeddings,  # (num_pts, embed_dim)
        k,
        eps=eps,
    )

    if LOG.isEnabledFor(logging.DEBUG):
        rc_basis = ops.bmm(
            coeffs.unsqueeze(1).astype(mindspore.float32),
            centered_embeddings[indices].astype(mindspore.float32),
        ).squeeze(1)
        for i in range(d):
            v_0 = U_d[:, i]
            v_1 = rc_basis[i]
            cos_sim = ops.cosine_similarity(  # pylint: disable=unexpected-keyword-arg
                v_0.unsqueeze(0), v_1.unsqueeze(0), axis=1
            ).squeeze()
            rms = ops.norm(v_0 - v_1)
            norm_rms = ops.norm(
                v_0 - (v_1 / ops.clamp(ops.norm(v_1), min=1e-6))
            )
            LOG.debug(
                f"Basis vector {i}: cos_sim = {cos_sim.asnumpy():.4f}, "
                f"RMS = {rms.asnumpy():.4f}, norm_rms = {norm_rms.asnumpy():.4f}"
            )

    return indices, coeffs


def compute_token_basis(  # pylint: disable=too-many-positional-arguments
    orig_embed: Tensor,
    donor_embed: Tensor,
    orig_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    junk_tokens: List[int],
    k: int,
) -> Tuple[Tensor, Tensor]:
    """Compute approximately orthogonal bases for both original and donor embeddings
    as sparse linear combinations of elements.

    Args:
        orig_embed: Original embedding matrix
        donor_embed: Donor embedding matrix
        orig_vocab: Vocabulary mapping for original model
        donor_vocab: Vocabulary mapping for donor model
        junk_tokens: List of junk token indices to exclude
        k: Number of points to select per basis vector
    Returns:
        donor_basis: Approximate orthogonal basis for donor model
        orig_basis: Approximate orthogonal basis for original model
    """
    common_vocab = set(orig_vocab.keys()) & set(donor_vocab.keys())
    junk_set = set(junk_tokens)
    common_vocab = [
        tok
        for tok in common_vocab
        if (tok not in donor_vocab or donor_vocab[tok] not in junk_set)
    ]
    effective_dim = min(orig_embed.shape[1], donor_embed.shape[1])
    orig_shared_embeds = orig_embed[
        Tensor([orig_vocab[t] for t in common_vocab], dtype=mindspore.int64)
    ]
    donor_shared_embeds = donor_embed[
        Tensor([donor_vocab[t] for t in common_vocab], dtype=mindspore.int64)
    ]
    if donor_embed.shape[1] < orig_embed.shape[1]:
        basis_src_embeds = donor_shared_embeds
        LOG.debug("Using donor embeds to compute token basis")
    else:
        basis_src_embeds = orig_shared_embeds
        LOG.debug("Using original embeds to compute token basis")
    LOG.debug(f"Basis dimension: {effective_dim}")
    tb_indices, tb_weights = sparse_linear_basis(
        basis_src_embeds,
        k=k,
        d=effective_dim,
    )
    donor_basis = (
        ops.bmm(
            tb_weights.unsqueeze(1).astype(mindspore.float32),
            donor_shared_embeds[tb_indices].astype(mindspore.float32),
        )
        .squeeze(1)
        .astype(donor_embed.dtype)
    )
    orig_basis = (
        ops.bmm(
            tb_weights.unsqueeze(1).astype(mindspore.float32),
            orig_shared_embeds[tb_indices].astype(mindspore.float32),
        )
        .squeeze(1)
        .astype(orig_embed.dtype)
    )
    return (donor_basis, orig_basis)
