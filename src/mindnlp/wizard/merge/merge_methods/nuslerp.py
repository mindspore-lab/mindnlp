# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import Any, Dict, List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from typing_extensions import override

from ..architecture.base import WeightInfo
from ..common import ImmutableMap, ModelReference
from ..dtype_policy import choose_work_dtype
from ..graph import Task
from ..safe_ops import safe_norm, safe_where
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class NuSlerpTask(Task[mindspore.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    row_wise: bool
    flatten: bool
    base_model: Optional[ModelReference]
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, mindspore.Tensor]) -> mindspore.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        if self.base_model is not None:
            if len(tensors) != 3:
                raise RuntimeError(
                    "NuSlerp base model can not be one of the two models to merge"
                )
            base_tensor = tensors.pop(self.base_model)
        else:
            base_tensor = None

        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        if len(tensors) != 2:
            raise RuntimeError(
                "NuSlerp merge expects exactly two models (plus optional base model)"
            )

        if abs(sum(weights)) < 1e-6:
            t = 0.5
        else:
            t = weights[1] / sum(weights)

        lhs, rhs = tensors[0], tensors[1]
        threshold = self.max_tensor_mem_gb
        if (
            threshold is not None
            and self.split_pieces > 1
            and lhs.ndim >= 1
            and int(lhs.nbytes) > int(float(threshold) * (1024**3))
            and int(lhs.shape[0]) >= self.split_pieces
        ):
            total = int(lhs.shape[0])
            outputs = []
            for piece_idx in range(self.split_pieces):
                start = (total * piece_idx) // self.split_pieces
                end = (total * (piece_idx + 1)) // self.split_pieces
                if end <= start:
                    continue
                base_piece = base_tensor[start:end] if base_tensor is not None else None
                outputs.append(self._merge_core(lhs[start:end], rhs[start:end], t, base_piece))
            if outputs:
                return ops.concat(outputs, axis=0)
        return self._merge_core(lhs, rhs, t, base_tensor)

    def _merge_core(
        self,
        lhs: mindspore.Tensor,
        rhs: mindspore.Tensor,
        t: float,
        base_tensor: Optional[mindspore.Tensor],
    ) -> mindspore.Tensor:
        prepped = [lhs, rhs]
        if base_tensor is not None:
            prepped.append(base_tensor)
        rectify_embed_sizes(self.weight_info, prepped)
        if base_tensor is not None:
            base = prepped[2]
            return base + nuslerp(
                t,
                prepped[0] - base,
                prepped[1] - base,
                dim=0 if self.row_wise else -1,
                flatten=self.flatten,
            )
        return nuslerp(
            t,
            prepped[0],
            prepped[1],
            dim=0 if self.row_wise else -1,
            flatten=self.flatten,
        )


class NuSlerpMerge(MergeMethod):
    def name(self) -> str:
        return "nuslerp"

    @override
    def pretty_name(self):
        return "NuSLERP"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="nuslerp_row_wise",
                required=False,
                default_value=False,
            ),
            ConfigParameterDef(
                name="nuslerp_flatten",
                required=False,
                default_value=True,
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **_kwargs,
    ) -> Task:
        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            row_wise=parameters["nuslerp_row_wise"],
            flatten=parameters["nuslerp_flatten"],
            base_model=base_model,
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )


def nuslerp(  # pylint: disable=too-many-positional-arguments
    t: float,
    v0: mindspore.Tensor,
    v1: mindspore.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    flatten: bool = False,
):
    out_dtype = v0.dtype
    work_dtype = choose_work_dtype(out_dtype)
    v0 = v0.astype(work_dtype)
    v1 = v1.astype(work_dtype)
    out_shape = v0.shape

    def _normalize(x: mindspore.Tensor, eps: float = 1e-7) -> mindspore.Tensor:
        return x / safe_norm(
            x, axis=-1, keepdims=True, out_dtype=mindspore.float32, op_name="nuslerp.norm"
        ).clamp(min=eps)

    if flatten:
        v0 = v0.reshape(-1)
        v1 = v1.reshape(-1)
    elif dim != -1:
        v0 = v0.transpose(dim, -1)
        v1 = v1.transpose(dim, -1)

    v0_u = _normalize(v0)
    v1_u = _normalize(v1)

    cos_theta = (v0_u * v1_u).sum(axis=-1, keepdims=True)
    theta = ops.acos(cos_theta.clamp(-1, 1))
    sin_theta = ops.sin(theta)

    colinear = sin_theta.abs() < eps
    res = (ops.sin((1 - t) * theta) * v0 + ops.sin(t * theta) * v1) / sin_theta
    fallback = (1 - t) * v0 + t * v1
    res = safe_where(colinear, fallback, res, out_dtype=work_dtype, op_name="nuslerp.where")

    if dim != -1 and not flatten:
        res = res.transpose(dim, -1)
    return res.reshape(out_shape).astype(out_dtype)
