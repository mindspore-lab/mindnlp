# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from enum import Enum
from typing import Optional

import mindspore
from mindspore import ops
import numpy as np

from .safe_ops import safe_abs, safe_norm, safe_sum


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    magnitude_outliers = "magnitude_outliers"
    della_magprune = "della_magprune"


class RescaleNorm(str, Enum):
    l1 = "l1"
    l2 = "l2"
    linf = "linf"


def rescaled_masked_tensor(
    tensor: mindspore.Tensor,
    mask: mindspore.Tensor,
    norm: Optional[RescaleNorm],
    eps: float = 1e-7,
) -> mindspore.Tensor:
    """Apply a mask to a tensor and rescale to match the original tensor norm.

    Returns the result in ``work_dtype`` (float32 on CPU) so that callers
    can chain further arithmetic without intermediate half-precision overflow.
    The caller is responsible for the final cast back to the output dtype.

    Args:
        tensor (mindspore.Tensor): Input tensor.
        mask (mindspore.Tensor): Mask to apply.
        norm (RescaleNorm): Which norm to match (l1, l2, linf).
        eps (float): Tolerance for small norms to avoid division by zero.
    """
    work_dtype = mindspore.float32
    tensor_work = tensor.astype(work_dtype)
    mask_work = mask.astype(work_dtype)
    masked = ops.mul(tensor_work, mask_work)
    if norm is None:
        return masked
    elif norm == RescaleNorm.l1:
        before_scale = safe_sum(
            safe_abs(tensor_work, out_dtype=work_dtype, op_name="sparsify.l1_before_abs"),
            out_dtype=work_dtype,
            op_name="sparsify.l1_before_sum",
        )
        after_scale = safe_sum(
            safe_abs(masked, out_dtype=work_dtype, op_name="sparsify.l1_after_abs"),
            out_dtype=work_dtype,
            op_name="sparsify.l1_after_sum",
        )
    elif norm == RescaleNorm.l2:
        before_scale = safe_norm(
            tensor_work, out_dtype=mindspore.float32, op_name="sparsify.l2_before"
        )
        after_scale = safe_norm(
            masked, out_dtype=mindspore.float32, op_name="sparsify.l2_after"
        )
    elif norm == RescaleNorm.linf:
        before_scale = ops.max(
            safe_abs(tensor_work, out_dtype=work_dtype, op_name="sparsify.linf_before_abs")
        )[0]
        after_scale = ops.max(
            safe_abs(masked, out_dtype=work_dtype, op_name="sparsify.linf_after_abs")
        )[0]
    else:
        raise NotImplementedError(norm)
    before_v = float(before_scale.astype(mindspore.float32).asnumpy().item())
    after_v = float(after_scale.astype(mindspore.float32).asnumpy().item())
    if before_v < eps or after_v < eps:
        return masked
    return ops.mul(masked, ops.div(before_scale, after_scale))


def magnitude(
    tensor: mindspore.Tensor, density: float, rescale_norm: Optional[RescaleNorm] = None
) -> mindspore.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.numel())
    assert k > 0, "not gonna zero out the whole tensor buddy"

    w = tensor.astype(mindspore.float32).abs().reshape(-1)
    topk_indices = ops.argsort(w, descending=True)[:k].asnumpy()

    mask_np = np.zeros(tensor.numel(), dtype=np.float32)
    mask_np[topk_indices] = 1.0
    mask = mindspore.Tensor(mask_np).reshape(tensor.shape)

    return rescaled_masked_tensor(tensor, mask, rescale_norm)


def magnitude_outliers(
    tensor: mindspore.Tensor,
    density: float,
    rescale_norm: Optional[RescaleNorm] = None,
    gamma: float = 0.01,
):
    """Masks out smallest values in addition to large outliers.

    The `gamma` proportion of the largest weights are first removed, then the
    smallest weights are removed to achieve the desired density.

    Args:
        tensor (mindspore.Tensor): The tensor to sparsify.
        density (float): The proportion of weights to retain.
        gamma (float): Percent of largest weights to remove.
    """
    if density >= 1:
        return tensor

    num_elems = tensor.numel()
    target_n = int(density * num_elems)
    n_top = int(gamma * num_elems)
    n_bot = num_elems - target_n - n_top

    if n_bot < 0:
        # cut down on the number of large weights to remove in
        # order to hit the target density
        n_top += n_bot
        n_bot = 0

    w = tensor.astype(mindspore.float32).abs().reshape(-1)
    indices = ops.sort(w, descending=False)[1].asnumpy()

    mask_np = np.zeros(tensor.numel(), dtype=np.float32)
    if n_top > 0:
        keep = indices[n_bot:-n_top]
    else:
        keep = indices[n_bot:]
    mask_np[keep] = 1.0
    mask = mindspore.Tensor(mask_np).reshape(tensor.shape)

    return rescaled_masked_tensor(tensor, mask, rescale_norm)


def bernoulli(
    tensor: mindspore.Tensor, density: float, rescale_norm: Optional[RescaleNorm] = None
) -> mindspore.Tensor:
    if density >= 1:
        return tensor

    work_dtype = mindspore.float32
    probs = ops.full(tensor.shape, density, dtype=mindspore.float32)
    rand = mindspore.Tensor(np.random.rand(*tensor.shape).astype(np.float32))
    mask = (rand < probs).astype(work_dtype)
    return rescaled_masked_tensor(tensor.astype(work_dtype), mask, rescale_norm)


def della_magprune(
    tensor: mindspore.Tensor,
    density: float,
    epsilon: float,
    rescale_norm: Optional[RescaleNorm] = None,
) -> mindspore.Tensor:
    if density >= 1:
        return tensor
    if density <= 0:
        return ops.zeros_like(tensor)
    orig_shape = tensor.shape

    if density + epsilon >= 1 or density - epsilon <= 0:
        raise ValueError(
            "Epsilon must be chosen such that density +/- epsilon is in (0, 1)"
        )

    work_dtype = mindspore.float32

    if len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    magnitudes = safe_abs(
        tensor, out_dtype=mindspore.float32, op_name="sparsify.della.abs"
    )

    sorted_indices = ops.argsort(magnitudes, axis=1, descending=False)
    ranks = ops.argsort(sorted_indices, axis=1).astype(work_dtype) + 1

    min_ranks, _ = ops.min(ranks, 1, True)
    max_ranks, _ = ops.max(ranks, 1, True)
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = ((density - epsilon) + rank_norm * 2 * epsilon).astype(mindspore.float32)
    # MindSpore Ascend backend may lack a Bernoulli adapter on some versions.
    # Use an equivalent uniform-sampling implementation instead.
    rand = mindspore.Tensor(np.random.rand(*probs.shape).astype(np.float32))
    mask = (rand < probs).astype(work_dtype)

    return rescaled_masked_tensor(tensor.astype(work_dtype), mask, rescale_norm).reshape(orig_shape)


def sparsify(  # pylint: disable=too-many-positional-arguments
    tensor: mindspore.Tensor,
    density: float,
    method: SparsificationMethod,
    gamma: float = 0,
    epsilon: float = 0,
    rescale_norm: Optional[RescaleNorm] = None,
) -> mindspore.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale_norm=rescale_norm)
    elif method == SparsificationMethod.magnitude_outliers:
        return magnitude_outliers(
            tensor, density=density, rescale_norm=rescale_norm, gamma=gamma
        )
    elif method == SparsificationMethod.della_magprune:
        return della_magprune(
            tensor, density=density, epsilon=epsilon, rescale_norm=rescale_norm
        )
    else:
        raise NotImplementedError(method)
