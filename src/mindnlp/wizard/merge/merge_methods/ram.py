# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import List, Tuple

import mindspore  # pylint: disable=import-error

from ..dtype_policy import choose_work_dtype
from ..safe_ops import safe_stack
from .easy_define import merge_method


@merge_method(
    name="ram",
    pretty_name="Reinforced Agent Merging",
    reference_url="https://arxiv.org/abs/2601.13572",
)
def ram_merge(
    tensors: List[mindspore.Tensor],
    base_tensor: mindspore.Tensor,
    epsilon: float = 1e-5,
) -> mindspore.Tensor:
    if not tensors:
        return base_tensor
    work_dtype = choose_work_dtype(base_tensor.dtype)
    base_work = base_tensor.astype(work_dtype)
    tensors_work = [t.astype(work_dtype) for t in tensors]

    (
        tv_flat,
        nonzero_mask,
        contrib_counts,
        overlap_mask,
        unique_mask,
    ) = _prepare_ram_vectors(tensors_work, base_work, epsilon)

    tv_flat_z = tv_flat * nonzero_mask
    merged_tv_flat = (
        (tv_flat_z * unique_mask)
        + (tv_flat_z * overlap_mask / contrib_counts.clamp(min=1))
    ).sum(axis=0, keepdims=True)

    return (base_work + merged_tv_flat.reshape(base_tensor.shape)).astype(
        base_tensor.dtype
    )


@merge_method(
    name="ramplus_tl",
    pretty_name="Reinforced Agent Merging Plus (Tensor-Local)",
    reference_url="https://arxiv.org/abs/2601.13572",
)
def ramplus_tl_merge(
    tensors: List[mindspore.Tensor],
    base_tensor: mindspore.Tensor,
    r: float = 0.1,
    alpha: float = 0.2,
    epsilon: float = 1e-5,
) -> mindspore.Tensor:
    if not tensors:
        return base_tensor
    work_dtype = choose_work_dtype(base_tensor.dtype)
    base_work = base_tensor.astype(work_dtype)
    tensors_work = [t.astype(work_dtype) for t in tensors]

    (
        tv_flat,
        nonzero_mask,
        contrib_counts,
        overlap_mask,
        unique_mask,
    ) = _prepare_ram_vectors(tensors_work, base_work, epsilon)

    shared_counts = (nonzero_mask & overlap_mask).sum(axis=1)
    unique_counts = (nonzero_mask & unique_mask).sum(axis=1)
    rho = shared_counts / unique_counts.clamp(min=epsilon)
    lambda_ = 1 + r * rho.clamp(min=0, max=alpha)

    tv_flat_z = tv_flat * nonzero_mask
    merged_tv_flat = (
        (tv_flat_z * unique_mask * lambda_.unsqueeze(-1))
        + (tv_flat_z * overlap_mask / contrib_counts.clamp(min=1))
    ).sum(axis=0, keepdims=True)
    merged_tv = merged_tv_flat.reshape(base_tensor.shape)
    return (base_work + merged_tv).astype(base_tensor.dtype)


def _prepare_ram_vectors(
    tensors: List[mindspore.Tensor], base_tensor: mindspore.Tensor, epsilon: float
) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
    """
    Helper function to compute task vectors, masks, and counts shared by RAM methods.
    Returns:
        tv_flat: Flattened task vectors
        nonzero_mask: Mask of values > epsilon
        contrib_counts: Count of models contributing to each parameter
        overlap_mask: Mask where counts > 1
        unique_mask: Mask where counts == 1
    """
    task_vectors = safe_stack(
        [t - base_tensor for t in tensors],
        axis=0,
        out_dtype=base_tensor.dtype,
        op_name="ram.stack",
    )
    tv_flat = task_vectors.reshape(len(tensors), -1)

    nonzero_mask = tv_flat.abs() > epsilon

    contrib_counts = nonzero_mask.sum(axis=0, keepdims=True)

    overlap_mask = contrib_counts > 1
    unique_mask = contrib_counts == 1

    return tv_flat, nonzero_mask, contrib_counts, overlap_mask, unique_mask
