# Copyright (c) MindNLP Wizard contributors.
# Licensed under the Apache License, Version 2.0.

"""Preflight checks for merge runtime safety."""

from __future__ import annotations

import logging


import mindspore
from mindspore import ops

from .common import dtype_from_name
from .config import MergeConfiguration
from .dtype_policy import choose_work_dtype
from .options import MergeOptions
from .safe_ops import safe_abs, safe_mul, safe_stack, safe_sum, safe_where

LOG = logging.getLogger(__name__)

_HALF_DTYPE_NAMES = {"bfloat16", "float16", "bf16", "fp16", "half"}
_METHODS_REQUIRING_HALF_PRECHECK = {
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_linear",
    "breadcrumbs",
    "breadcrumbs_ties",
    "della",
    "della_linear",
    "sce",
}


def run_merge_preflight(merge_config: MergeConfiguration, options: MergeOptions) -> None:
    """Run quick probes to fail fast on unsafe runtime combinations."""
    merge_method = (merge_config.merge_method or "").strip().lower()
    dtype_name = ((merge_config.out_dtype or merge_config.dtype or "")).strip().lower()
    if merge_method not in _METHODS_REQUIRING_HALF_PRECHECK:
        return
    if dtype_name not in _HALF_DTYPE_NAMES:
        return
    _probe_half_precision_math(merge_method=merge_method, dtype_name=dtype_name)


def _probe_half_precision_math(*, merge_method: str, dtype_name: str) -> None:
    target = "UNKNOWN"
    try:
        target = str(mindspore.get_context("device_target"))
    except Exception:
        pass

    test_dtype = dtype_from_name(dtype_name) or mindspore.float16
    work_dtype = choose_work_dtype(test_dtype)
    try:
        a = mindspore.Tensor([1.0, -2.0, 3.0], dtype=test_dtype)
        b = mindspore.Tensor([2.0, 4.0, -1.0], dtype=test_dtype)
        stacked = safe_stack([a, b], axis=0, out_dtype=work_dtype, op_name="preflight.stack")
        weights = mindspore.Tensor([[0.5], [0.5]], dtype=work_dtype)
        weighted = safe_mul(stacked, weights, out_dtype=work_dtype, op_name="preflight.mul")
        merged = safe_sum(weighted, axis=0, out_dtype=work_dtype, op_name="preflight.sum")

        sign = ops.sign(weighted)
        mask = ops.equal(
            sign,
            safe_where(
                ops.greater_equal(
                    safe_sum(sign, axis=0, out_dtype=work_dtype, op_name="preflight.sign_sum"),
                    mindspore.Tensor(0, dtype=work_dtype),
                ),
                mindspore.Tensor(1, dtype=work_dtype),
                mindspore.Tensor(-1, dtype=work_dtype),
                out_dtype=work_dtype,
                op_name="preflight.majority",
            ),
        )
        masked = safe_mul(weighted, mask, out_dtype=work_dtype, op_name="preflight.mask_mul")
        divisor = safe_sum(mask.astype(work_dtype), axis=0, out_dtype=work_dtype, op_name="preflight.divisor")
        divisor = safe_where(
            ops.less(safe_abs(divisor, out_dtype=work_dtype, op_name="preflight.abs"), mindspore.Tensor(1e-8, dtype=work_dtype)),
            mindspore.Tensor(1, dtype=work_dtype),
            divisor,
            out_dtype=work_dtype,
            op_name="preflight.divisor_fix",
        )
        _ = ops.add(merged, ops.div(safe_sum(masked, axis=0, out_dtype=work_dtype, op_name="preflight.mask_sum"), divisor))
    except Exception as exc:
        raise RuntimeError(
            "Wizard merge preflight failed for half precision runtime. "
            f"merge_method={merge_method}, dtype={dtype_name}, device_target={target}. "
            "This indicates an unsafe compute path before actual merge. "
            "Please inspect wizard safe_ops / generalized_task_arithmetic path."
        ) from exc

    LOG.info(
        "Wizard preflight passed: merge_method=%s dtype=%s device_target=%s",
        merge_method,
        dtype_name,
        target,
    )
