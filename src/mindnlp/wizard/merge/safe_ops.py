# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""Safe operator wrappers."""

from __future__ import annotations

from typing import Iterable, Optional, Union

import mindspore
from mindspore import ops

from .dtype_policy import (
    cast_back,
    cast_many_to_work,
    cast_to_work,
    choose_work_dtype,
    warn_safe_path_once,
)


def _resolve_work_dtype(
    ref_dtype: mindspore.dtype,
    *,
    op_name: str,
    out_dtype: Optional[mindspore.dtype],
    device_target: Optional[str],
) -> tuple[mindspore.dtype, mindspore.dtype]:
    result_dtype = out_dtype or ref_dtype
    work_dtype = choose_work_dtype(result_dtype, device_target=device_target)
    warn_safe_path_once(
        op_name,
        in_dtype=result_dtype,
        work_dtype=work_dtype,
        device_target=device_target,
    )
    return result_dtype, work_dtype


def safe_stack(
    tensors: Iterable[mindspore.Tensor],
    *,
    axis: int = 0,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "stack",
) -> mindspore.Tensor:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("safe_stack expects non-empty tensors")
    result_dtype, work_dtype = _resolve_work_dtype(
        tensors[0].dtype,
        op_name=op_name,
        out_dtype=out_dtype,
        device_target=device_target,
    )
    stacked = ops.stack(cast_many_to_work(tensors, work_dtype), axis=axis)
    return cast_back(stacked, result_dtype)


def safe_sum(
    tensor: mindspore.Tensor,
    *,
    axis=None,
    keepdims: bool = False,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "sum",
) -> mindspore.Tensor:
    result_dtype, work_dtype = _resolve_work_dtype(
        tensor.dtype,
        op_name=op_name,
        out_dtype=out_dtype,
        device_target=device_target,
    )
    reduced = ops.sum(cast_to_work(tensor, work_dtype), dim=axis, keepdim=keepdims)
    return cast_back(reduced, result_dtype)


def safe_mul(
    lhs: mindspore.Tensor,
    rhs: Union[mindspore.Tensor, float, int],
    *,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "mul",
) -> mindspore.Tensor:
    if not isinstance(rhs, mindspore.Tensor):
        rhs = mindspore.Tensor(rhs, dtype=lhs.dtype)
    result_dtype, work_dtype = _resolve_work_dtype(
        out_dtype or lhs.dtype,
        op_name=op_name,
        out_dtype=out_dtype or lhs.dtype,
        device_target=device_target,
    )
    res = ops.mul(cast_to_work(lhs, work_dtype), cast_to_work(rhs, work_dtype))
    return cast_back(res, result_dtype)


def safe_abs(
    tensor: mindspore.Tensor,
    *,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "abs",
) -> mindspore.Tensor:
    result_dtype, work_dtype = _resolve_work_dtype(
        out_dtype or tensor.dtype,
        op_name=op_name,
        out_dtype=out_dtype or tensor.dtype,
        device_target=device_target,
    )
    res = ops.abs(cast_to_work(tensor, work_dtype))
    return cast_back(res, result_dtype)


def safe_norm(
    tensor: mindspore.Tensor,
    *,
    axis=None,
    keepdims: bool = False,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "norm",
) -> mindspore.Tensor:
    result_dtype, work_dtype = _resolve_work_dtype(
        out_dtype or tensor.dtype,
        op_name=op_name,
        out_dtype=out_dtype or tensor.dtype,
        device_target=device_target,
    )
    res = ops.norm(
        cast_to_work(tensor, work_dtype),
        dim=axis,
        keepdim=keepdims,
    )
    return cast_back(res, result_dtype)


def safe_where(
    condition: mindspore.Tensor,
    x: mindspore.Tensor,
    y: mindspore.Tensor,
    *,
    out_dtype: Optional[mindspore.dtype] = None,
    device_target: Optional[str] = None,
    op_name: str = "where",
) -> mindspore.Tensor:
    result_dtype, work_dtype = _resolve_work_dtype(
        out_dtype or x.dtype,
        op_name=op_name,
        out_dtype=out_dtype or x.dtype,
        device_target=device_target,
    )
    res = ops.where(
        condition,
        cast_to_work(x, work_dtype),
        cast_to_work(y, work_dtype),
    )
    return cast_back(res, result_dtype)
