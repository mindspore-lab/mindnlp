# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
# Ported from MergeKit to MindSpore for Wizard

import enum
import logging
from typing import Optional, Tuple

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from mindspore import Tensor  # pylint: disable=import-error

LOG = logging.getLogger(__name__)


class DistanceMetric(enum.Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class WeightingScheme(enum.Enum):
    DISTANCE_PROPORTIONAL = "distance_proportional"
    BARYCENTRIC = "barycentric"
    LEAST_SQUARES = "least_squares"


def approximate_from_landmarks(
    targets: Tensor,
    points: Tensor,
    distances: Tensor,
    scheme: WeightingScheme = WeightingScheme.DISTANCE_PROPORTIONAL,
    cosine_similarity: bool = False,
) -> Tensor:
    batch_size, embedding_dim = targets.shape
    assert points.ndim == 3 and points.shape == (
        batch_size,
        points.shape[1],
        embedding_dim,
    )
    num_points = points.shape[1]
    assert points.shape[2] == embedding_dim
    assert distances.shape == (batch_size, num_points)

    if scheme == WeightingScheme.DISTANCE_PROPORTIONAL:
        if cosine_similarity:
            weights = 1 - distances
        else:
            weights = 1 / ops.clamp(distances, min=1e-6)
        weights = weights / ops.clamp(weights.sum(axis=1, keepdims=True), min=1e-6)
    elif scheme == WeightingScheme.BARYCENTRIC:
        weights = barycentric_weights(targets, points)
    elif scheme == WeightingScheme.LEAST_SQUARES:
        weights = ops.lstsq(
            points.swapaxes(1, 2).astype(mindspore.float32),
            targets.unsqueeze(-1).astype(mindspore.float32),
        ).squeeze(-1)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return weights


def barycentric_weights(targets: Tensor, points: Tensor) -> Tensor:
    batch_size, num_points, _embedding_dim = points.shape
    ptp = ops.bmm(points, points.swapaxes(1, 2))
    ones_col = ops.ones((batch_size, num_points, 1), dtype=points.dtype)
    ones_row = ops.ones((batch_size, 1, num_points), dtype=points.dtype)
    zeros = ops.zeros((batch_size, 1, 1), dtype=points.dtype)
    upper = ops.cat([ptp, ones_col], axis=2)
    lower = ops.cat([ones_row, zeros], axis=2)
    augmented_matrix = ops.cat([upper, lower], axis=1)
    rhs_upper = ops.bmm(targets.unsqueeze(1), points.swapaxes(1, 2)).squeeze(1)
    rhs_lower = ops.ones((batch_size, 1), dtype=points.dtype)
    rhs = ops.cat([rhs_upper, rhs_lower], axis=1)
    solution = ops.lstsq(
        augmented_matrix.astype(mindspore.float32),
        rhs.unsqueeze(-1).astype(mindspore.float32),
    ).squeeze(-1)
    return solution[..., :num_points]


def _cosine_sim(x1: Tensor, x2: Tensor, eps: float = 1e-6) -> Tensor:
    w1 = ops.norm(x1, ord=2, axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    w2 = ops.norm(x2, ord=2, axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
    return ops.matmul(x1, x2.T) / ops.clamp(w1 * w2.T, min=eps)


def common_interp_approximate(
    targets: Tensor,
    a_embeddings: Tensor,
    k: Optional[int] = None,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    weight_scheme: WeightingScheme = WeightingScheme.DISTANCE_PROPORTIONAL,
) -> Tuple[Tensor, Tensor]:
    assert targets.ndim == 2
    assert a_embeddings.ndim == 2
    assert targets.shape[1] == a_embeddings.shape[1]
    assert (k is None) or (k > 0), "k must be positive"

    if metric == DistanceMetric.EUCLIDEAN:
        distances = _cdist(targets, a_embeddings)
    elif metric == DistanceMetric.COSINE:
        distances = 1 - _cosine_sim(targets, a_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    if k is not None:
        _, indices = ops.topk(distances, k=k, dim=1, largest=False)
        knn_distances = ops.gather_elements(distances, 1, indices)
    else:
        indices = ops.arange(a_embeddings.shape[0]).expand(
            targets.shape[0], -1
        )
        knn_distances = distances

    weights = approximate_from_landmarks(
        targets,
        a_embeddings[indices],
        knn_distances,
        scheme=weight_scheme,
        cosine_similarity=metric == DistanceMetric.COSINE,
    )

    approx = (
        ops.bmm(
            weights.unsqueeze(1).astype(mindspore.float32),
            a_embeddings[indices].astype(mindspore.float32),
        )
        .squeeze(1)
        .astype(targets.dtype)
    )
    err = ops.norm(approx - targets, axis=1)  # pylint: disable=unexpected-keyword-arg
    LOG.debug(f"Reconstruction error: {err.mean()}")
    return indices, weights


def _cdist(x1: Tensor, x2: Tensor) -> Tensor:
    """Compute pairwise L2 distances between rows of x1 and x2."""
    x1_sq = (x1 ** 2).sum(axis=1, keepdims=True)
    x2_sq = (x2 ** 2).sum(axis=1, keepdims=True)
    cross = ops.matmul(x1, x2.T)
    dist_sq = x1_sq - 2 * cross + x2_sq.T
    return ops.sqrt(ops.clamp(dist_sq, min=0.0))
