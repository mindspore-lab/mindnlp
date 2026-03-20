# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from pydantic import BaseModel
from typing_extensions import Literal, override

from ..architecture.base import WeightInfo
from ..common import ImmutableMap, ModelReference
from ..graph import Task
from ..safe_ops import safe_abs, safe_mul, safe_stack, safe_sum, safe_where
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from ..sparsify import RescaleNorm, SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    def name(self) -> str:
        return self.method_name

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name

    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            res.append(
                ConfigParameterDef(
                    name="gamma",
                    default_value=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.della_magprune:
            res.append(
                ConfigParameterDef(
                    name="epsilon",
                    default_value=0.15,
                )
            )
        return res

    def make_task(  # pylint: disable=too-many-positional-arguments
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
    ) -> Task:
        return GTATask(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            lambda_=parameters["lambda"],
            rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
            weight_info=output_weight,
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )


class GTATask(Task[mindspore.Tensor]):
    method: GeneralizedTaskArithmeticMerge
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    lambda_: float
    rescale_norm: Optional[RescaleNorm]
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, mindspore.Tensor],
        **_kwargs,
    ) -> mindspore.Tensor:
        if not tensors:
            raise RuntimeError("No tensors provided to GTATask")

        first = next(iter(tensors.values()))
        threshold = self.max_tensor_mem_gb
        if (
            threshold is not None
            and self.split_pieces > 1
            and first.ndim >= 1
            and int(first.nbytes) > int(float(threshold) * (1024**3))
            and int(first.shape[0]) >= self.split_pieces
        ):
            total = int(first.shape[0])
            outputs = []
            for piece_idx in range(self.split_pieces):
                start = (total * piece_idx) // self.split_pieces
                end = (total * (piece_idx + 1)) // self.split_pieces
                if end <= start:
                    continue
                piece_tensors = {k: v[start:end] for k, v in tensors.items()}
                outputs.append(self._execute_core(piece_tensors))
            if outputs:
                return ops.concat(outputs, axis=0)

        return self._execute_core(tensors)

    def _execute_core(self, tensors: Dict[ModelReference, mindspore.Tensor]) -> mindspore.Tensor:
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            dict(tensors),
            tensor_parameters=self.tensor_parameters.data,
        )
        if not tvs:
            return base

        out_dtype = base.dtype
        work_dtype = mindspore.float32

        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]

                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]

                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    rescale_norm=self.rescale_norm,
                    **kwargs,
                )

        deltas = safe_stack(
            [tv["delta"] for tv in tvs],
            axis=0,
            out_dtype=work_dtype,
            op_name="gta.stack",
        )

        weight_list = [tv["weight"] for tv in tvs]
        w_np = np.array(weight_list, dtype=np.float32)
        for _ in range(len(deltas.shape) - 1):
            w_np = np.expand_dims(w_np, axis=-1)
        weights = mindspore.Tensor(w_np, dtype=work_dtype)

        weighted_deltas = safe_mul(
            deltas, weights, out_dtype=work_dtype, op_name="gta.mul"
        )

        if self.method.consensus_method:
            mask_dtype = mindspore.int8 if self.int8_mask else work_dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )

            mixed_delta = safe_sum(
                safe_mul(
                    weighted_deltas, mask, out_dtype=work_dtype, op_name="gta.mask_mul"
                ),
                axis=0,
                out_dtype=work_dtype,
                op_name="gta.mask_sum",
            )
            divisor = safe_sum(
                safe_mul(weights, mask, out_dtype=work_dtype, op_name="gta.div_mul"),
                axis=0,
                out_dtype=work_dtype,
                op_name="gta.div_sum",
            )
            zero_fill = mindspore.Tensor(1.0, dtype=work_dtype)
            divisor = safe_where(
                ops.equal(divisor, 0),
                zero_fill,
                divisor,
                out_dtype=work_dtype,
                op_name="gta.divisor_fill_zero",
            )
        else:
            mixed_delta = safe_sum(
                weighted_deltas, axis=0, out_dtype=work_dtype, op_name="gta.sum"
            )
            divisor = safe_sum(weights, axis=0, out_dtype=work_dtype, op_name="gta.wsum")
            tiny = mindspore.Tensor(1e-8, dtype=work_dtype)
            one = mindspore.Tensor(1.0, dtype=work_dtype)
            divisor = safe_where(
                ops.less(safe_abs(divisor, out_dtype=work_dtype, op_name="gta.divisor_abs"), tiny),
                one,
                divisor,
                out_dtype=work_dtype,
                op_name="gta.divisor_fill_tiny",
            )

        if self.normalize:
            mixed_delta = ops.div(mixed_delta, divisor)

        if self.lambda_ != 1:
            lambda_tensor = mindspore.Tensor(float(self.lambda_), dtype=work_dtype)
            mixed_delta = ops.mul(mixed_delta, lambda_tensor)

        result = ops.add(base.astype(work_dtype), mixed_delta)

        return result.astype(out_dtype)

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, mindspore.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
) -> Tuple[List[Dict[str, Any]], mindspore.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    parameter_name = weight_info.name

    # Compute deltas in float32 unconditionally to avoid unsupported
    # half-precision arithmetic and intermediate overflow.
    work_dtype = mindspore.float32
    base_work = base.astype(work_dtype)

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].astype(base.dtype)
        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x.astype(work_dtype) - base_work
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base


def get_mask(
    delta: mindspore.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[mindspore.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = ops.sign(delta).astype(mask_dtype)

    # Use explicit typed constants to avoid MindSpore CPU rejecting
    # mixed-type Mul/Sub (e.g. int8 × int64).
    _two = mindspore.Tensor(2, dtype=mask_dtype)
    _one = mindspore.Tensor(1, dtype=mask_dtype)

    if method == "sum":
        sign_weight = safe_sum(delta, axis=0, out_dtype=delta.dtype, op_name="gta.mask_sign_sum")
        ge_zero = ops.greater_equal(sign_weight, mindspore.Tensor(0, dtype=sign_weight.dtype))
        majority_sign = ops.sub(ops.mul(ge_zero.astype(mask_dtype), _two), _one)
        del sign_weight
    elif method == "count":
        sign_sum = safe_sum(sign, axis=0, out_dtype=sign.dtype, op_name="gta.mask_count_sum")
        ge_zero = ops.greater_equal(sign_sum, mindspore.Tensor(0, dtype=sign_sum.dtype))
        majority_sign = ops.sub(ops.mul(ge_zero.astype(mask_dtype), _two), _one)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return ops.equal(sign, majority_sign)
