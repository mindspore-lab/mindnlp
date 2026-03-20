# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error

from ..dtype_policy import choose_work_dtype
from ..safe_ops import safe_norm, safe_stack
from .easy_define import merge_method


@merge_method(
    name="multislerp",
    pretty_name="Multi-SLERP",
    reference_url="https://goddard.blog/posts/multislerp-wow-what-a-cool-idea",
)
def multislerp(
    tensors: List[mindspore.Tensor],
    weight: List[float],
    base_tensor: Optional[mindspore.Tensor] = None,
    normalize_weights: bool = True,
    eps: float = 1e-8,
):
    """
    Implements barycentric interpolation on a hypersphere.

    The approach:
    1. Project points onto a tangent space at their weighted Euclidean mean.
    2. Perform the interpolation in the tangent space.
    3. Project the result back to the hypersphere.

    Limitations:
    - The weighted sum of the input tensors must not be zero.
    - The tensors must not be all parallel or antiparallel.

    Args:
        tensors: List of tensors to interpolate
        weight: List of weights for each tensor
        base_tensor: Optional tensor defining the origin of the hypersphere
        normalize_weights: If True, the weights will be normalized to sum to 1
        eps: Small constant for numerical stability
    """
    if len(tensors) == 1:
        return tensors[0]

    out_dtype = tensors[0].dtype
    work_dtype = choose_work_dtype(out_dtype)
    tensors = safe_stack(tensors, axis=0, out_dtype=work_dtype, op_name="multislerp.stack")
    if base_tensor is not None:
        tensors = tensors - base_tensor.astype(work_dtype)

    tensors_flat = tensors.reshape(tensors.shape[0], -1)

    weights = mindspore.Tensor(weight, dtype=work_dtype)
    if normalize_weights:
        weights = weights / weights.sum()

    norms = safe_norm(
        tensors_flat,
        axis=-1,
        keepdims=True,
        out_dtype=mindspore.float32,
        op_name="multislerp.norm",
    )
    unit_tensors = tensors_flat / (norms + eps)

    mean = (unit_tensors * weights.reshape(-1, 1)).sum(0)
    mean_norm = safe_norm(mean, out_dtype=mindspore.float32, op_name="multislerp.mean_norm")
    if mean_norm < eps:
        if tensors.shape[0] == 2:
            res = (tensors[0] * weights[0] + tensors[1] * weights[1]).reshape(
                tensors.shape[1:]
            )
            if base_tensor is not None:
                res = res + base_tensor.astype(work_dtype)
            return res.astype(out_dtype)
        raise ValueError(
            "The weighted sum of the input tensors is zero. This occurs when "
            "antipodal vectors or sets of vectors have weights that exactly "
            "balance out (e.g., vectors a,-a with equal weights). Try using "
            "different weights if you have antipodal vectors."
        )
    mean = mean / mean_norm

    dots = (unit_tensors * mean).sum(-1, keepdims=True)
    tangent_vectors = unit_tensors - dots * mean

    tangent_result = (tangent_vectors * weights.reshape(-1, 1)).sum(0)

    tangent_norm = safe_norm(
        tangent_result, out_dtype=mindspore.float32, op_name="multislerp.tangent_norm"
    ) + eps
    result = mean * ops.cos(tangent_norm) + tangent_result * (
        ops.sin(tangent_norm) / tangent_norm
    )

    avg_norm = (norms.squeeze(-1) * weights).sum()
    result = result * avg_norm
    result = result.reshape(tensors.shape[1:])

    if base_tensor is not None:
        result = result + base_tensor.astype(work_dtype)

    return result.astype(out_dtype)
