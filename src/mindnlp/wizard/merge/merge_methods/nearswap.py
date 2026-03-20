# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import List

import mindspore  # pylint: disable=import-error

from ..dtype_policy import choose_work_dtype
from ..safe_ops import safe_abs
from .easy_define import merge_method


@merge_method(
    name="nearswap",
    pretty_name="NearSwap",
    reference_url="https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001",
)
def nearswap_merge(
    tensors: List[mindspore.Tensor], base_tensor: mindspore.Tensor, t: float
) -> mindspore.Tensor:
    if not tensors:
        return base_tensor
    if len(tensors) != 1:
        raise RuntimeError(
            "NearSwap merge expects exactly two models, one base and one other"
        )
    out_dtype = base_tensor.dtype
    work_dtype = choose_work_dtype(out_dtype)
    a = base_tensor.astype(work_dtype)
    b = tensors[0].astype(work_dtype)

    absdiff = safe_abs(a - b, out_dtype=work_dtype, op_name="nearswap.absdiff")
    weight = (t / absdiff.clamp(min=1e-6)).clamp(min=0, max=1)
    return (weight * b + (1 - weight) * a).astype(out_dtype)
