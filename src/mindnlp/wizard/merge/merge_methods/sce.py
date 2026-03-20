# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
import numpy as np

from ..dtype_policy import choose_work_dtype
from ..safe_ops import safe_mul, safe_stack, safe_sum
from .easy_define import merge_method
from .generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)


@merge_method(
    name="sce",
    pretty_name="SCE",
    reference_url="https://arxiv.org/abs/2408.07990",
)
def sce_merge(
    tensors: List[mindspore.Tensor],
    base_tensor: mindspore.Tensor,
    int8_mask: bool = False,
    select_topk: float = 1.0,
) -> mindspore.Tensor:
    if not tensors:
        return base_tensor
    work_dtype = choose_work_dtype(base_tensor.dtype)
    base_work = base_tensor.astype(work_dtype)
    mask_dtype = mindspore.int8 if int8_mask else work_dtype
    task_vectors = safe_stack(
        [t.astype(work_dtype) - base_work for t in tensors],
        axis=0,
        out_dtype=work_dtype,
        op_name="sce.stack",
    )

    if select_topk < 1:
        mask = sce_mask(task_vectors, select_topk, mask_dtype)
        task_vectors = safe_mul(
            task_vectors, mask.unsqueeze(0), out_dtype=work_dtype, op_name="sce.mask_mul"
        )

    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

    tv_weights = sce_weight(task_vectors)
    while tv_weights.ndim < task_vectors.ndim:
        tv_weights = tv_weights.unsqueeze(-1)

    erased_weights = safe_mul(
        tv_weights, erase_mask, out_dtype=work_dtype, op_name="sce.erase_mul"
    )
    merged_tv = safe_sum(
        safe_mul(task_vectors, erased_weights, out_dtype=work_dtype, op_name="sce.tv_mul"),
        axis=0,
        out_dtype=work_dtype,
        op_name="sce.tv_sum",
    )
    final_tv = merged_tv / safe_sum(
        erased_weights, axis=0, out_dtype=work_dtype, op_name="sce.weight_sum"
    ).clamp(min=1e-6)

    return (base_work + final_tv).astype(base_tensor.dtype)


def sce_weight(tvs: mindspore.Tensor) -> mindspore.Tensor:
    weights = ops.mean((tvs.astype(mindspore.float32)) ** 2, axis=list(range(1, tvs.ndim)))
    weight_sum = float(ops.sum(weights).asnumpy().item())
    if abs(weight_sum) < 1e-6:
        return ops.ones_like(weights) / weights.shape[0]
    return weights / weight_sum


def sce_mask(
    tvs: mindspore.Tensor, density: float, mask_dtype: Optional[mindspore.dtype] = None
):
    if density <= 0:
        return ops.zeros_like(tvs, dtype=mask_dtype)
    if density >= 1:
        return ops.ones_like(tvs, dtype=mask_dtype)

    var = ops.var(tvs.astype(mindspore.float32), axis=0, ddof=0)
    nonzero = int((var != 0).sum().asnumpy().item())
    k = int(nonzero * density)
    if k == 0:
        return ops.zeros_like(tvs, dtype=mask_dtype)

    _, indices = ops.topk(var.abs().reshape(-1), k=k, largest=True)
    mask_np = np.zeros(var.shape, dtype=np.float32)
    mask_np.reshape(-1)[indices.asnumpy()] = 1.0
    return mindspore.Tensor(mask_np, dtype=mask_dtype)
