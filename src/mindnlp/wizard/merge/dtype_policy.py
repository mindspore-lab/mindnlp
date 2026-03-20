# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""Dtype policy helpers for merge execution."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import mindspore
import numpy

LOG = logging.getLogger(__name__)

HALF_DTYPES = (mindspore.bfloat16, mindspore.float16)
_WARNED_SAFE_PATHS: Dict[str, bool] = {}


def get_device_target(device_target: Optional[str] = None) -> str:
    if device_target:
        return str(device_target).strip().upper()
    try:
        target = mindspore.get_context("device_target")
        if target:
            return str(target).strip().upper()
    except Exception:
        pass
    return "CPU"


def needs_safe_path(
    ref_dtype: mindspore.dtype,
    *,
    device_target: Optional[str] = None,
    promote_half_to_fp32: bool = True,
) -> bool:
    if not promote_half_to_fp32:
        return False
    return (
        get_device_target(device_target) == "CPU"
        and ref_dtype in HALF_DTYPES
    )


def choose_work_dtype(
    ref_dtype: mindspore.dtype,
    *,
    device_target: Optional[str] = None,
    promote_half_to_fp32: bool = True,
) -> mindspore.dtype:
    if needs_safe_path(
        ref_dtype,
        device_target=device_target,
        promote_half_to_fp32=promote_half_to_fp32,
    ):
        return mindspore.float32
    return ref_dtype


def cast_to_work(tensor: mindspore.Tensor, work_dtype: mindspore.dtype) -> mindspore.Tensor:
    if tensor.dtype == work_dtype:
        return tensor
    return tensor.astype(work_dtype)


def cast_back(tensor: mindspore.Tensor, out_dtype: mindspore.dtype) -> mindspore.Tensor:
    if tensor.dtype == out_dtype:
        return tensor
    return tensor.astype(out_dtype)


def cast_many_to_work(
    tensors: Iterable[mindspore.Tensor],
    work_dtype: mindspore.dtype,
) -> list[mindspore.Tensor]:
    return [cast_to_work(t, work_dtype) for t in tensors]


def warn_safe_path_once(
    op_name: str,
    *,
    in_dtype: mindspore.dtype,
    work_dtype: mindspore.dtype,
    device_target: Optional[str] = None,
) -> None:
    if in_dtype == work_dtype:
        return
    key = f"{get_device_target(device_target)}::{op_name}::{in_dtype}->{work_dtype}"
    if _WARNED_SAFE_PATHS.get(key):
        return
    _WARNED_SAFE_PATHS[key] = True
    LOG.warning(
        "Safe dtype path enabled for %s: %s -> %s on %s",
        op_name,
        in_dtype,
        work_dtype,
        get_device_target(device_target),
    )


# ---------------------------------------------------------------------------
# Centralised numpy <-> MindSpore conversion with bfloat16 safety
# ---------------------------------------------------------------------------

def _is_ml_dtypes_bfloat16(arr: numpy.ndarray) -> bool:
    """Check whether *arr* uses ``ml_dtypes.bfloat16``."""
    try:
        import ml_dtypes
        return arr.dtype == ml_dtypes.bfloat16
    except (ImportError, AttributeError):
        return False


def numpy_to_mindspore(arr: numpy.ndarray) -> mindspore.Tensor:
    """Convert a numpy array to a MindSpore tensor, handling ``bfloat16``.

    ``ml_dtypes.bfloat16`` is not a built-in numpy dtype, so
    ``mindspore.Tensor(arr)`` may silently misinterpret the buffer.
    This helper detects the dtype and passes an explicit
    ``dtype=mindspore.bfloat16`` when needed.
    """
    if _is_ml_dtypes_bfloat16(arr):
        return mindspore.Tensor(arr, dtype=mindspore.bfloat16)
    return mindspore.Tensor(arr)


def mindspore_to_numpy(tensor: mindspore.Tensor) -> numpy.ndarray:
    """Convert a MindSpore tensor to a numpy array, handling ``bfloat16``.

    MindSpore's ``asnumpy()`` for bfloat16 tensors may return a raw
    ``uint16`` view.  This helper reinterprets such arrays as
    ``ml_dtypes.bfloat16`` so that downstream consumers (safetensors,
    etc.) see the correct dtype and metadata.
    """
    arr = tensor.asnumpy()
    if tensor.dtype == mindspore.bfloat16:
        try:
            import ml_dtypes
            if arr.dtype != ml_dtypes.bfloat16:
                arr = arr.view(ml_dtypes.bfloat16)
        except ImportError:
            LOG.warning(
                "ml_dtypes is not installed — bfloat16 tensor will be "
                "exported as raw %s; downstream tools may misinterpret it.",
                arr.dtype,
            )
    return arr
