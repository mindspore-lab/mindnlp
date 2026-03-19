# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import logging

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error

LOG = logging.getLogger(__name__)


def landmark_pca_approximate(
    targets: Tensor,
    points_a: Tensor,
    points_b: Tensor,
) -> Tensor:
    """Given target points in space a and a set of reference points in both space a and b,
    approximate the target points in space b."""
    num_points, d_a = points_a.shape
    batch_size, _ = targets.shape
    _, d_b = points_b.shape
    assert (
        points_a.shape[0] == points_b.shape[0]
    ), "Number of points in A and B must match"
    assert targets.shape == (batch_size, d_a)

    effective_dim = min(d_a, d_b)

    out_dtype = targets.dtype
    points_a = points_a.astype(mindspore.float32)
    points_b = points_b.astype(mindspore.float32)
    targets = targets.astype(mindspore.float32)

    mean_a = points_a.mean(axis=0, keepdims=True)  # (1, D_a)
    mean_b = points_b.mean(axis=0, keepdims=True)  # (1, D_b)
    centered_a = points_a - mean_a  # (N, D_a)
    centered_b = points_b - mean_b  # (N, D_b)
    centered_targets = targets - mean_a  # (B, D_a)

    V_a = _pca_lowrank(centered_a, q=effective_dim)  # (D_a, effective_dim)
    V_b = _pca_lowrank(centered_b, q=effective_dim)  # (D_b, effective_dim)

    A_pca = ops.matmul(centered_a, V_a)  # (N, effective_dim)
    B_pca = ops.matmul(centered_b, V_b)  # (N, effective_dim)

    M = ops.matmul(B_pca.T, A_pca)  # (effective_dim, effective_dim)
    U, S, V = ops.svd(M)
    R = ops.matmul(U, V)  # (effective_dim, effective_dim)

    projected_a = ops.matmul(centered_targets, V_a)  # (B, effective_dim)
    rotated = ops.matmul(projected_a, R)  # (B, effective_dim)
    projected_b = ops.matmul(rotated, V_b.T)  # (B, D_b)

    approximated_b = projected_b + mean_b
    return approximated_b.astype(out_dtype)


def _pca_lowrank(data: Tensor, q: int) -> Tensor:
    """Compute the top-q right singular vectors (V) of the data matrix via SVD.

    This is a simplified replacement for torch.pca_lowrank, returning only V.
    """
    U, S, V = ops.svd(data)
    return V[:, :q]
