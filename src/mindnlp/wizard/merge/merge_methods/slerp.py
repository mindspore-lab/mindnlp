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
from ..graph import Task
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class SlerpTask(Task[mindspore.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    t: float
    weight_info: WeightInfo
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, mindspore.Tensor]) -> mindspore.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError("Slerp merge expects exactly two models")
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        lhs, rhs = a[1], b[1]
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
                outputs.append(self._merge_core(lhs[start:end], rhs[start:end]))
            if outputs:
                return ops.concat(outputs, axis=0)
        return self._merge_core(lhs, rhs)

    def _merge_core(
        self, lhs: mindspore.Tensor, rhs: mindspore.Tensor
    ) -> mindspore.Tensor:
        prepped_tensors = [lhs, rhs]
        rectify_embed_sizes(self.weight_info, prepped_tensors)
        return slerp(self.t, prepped_tensors[0], prepped_tensors[1]).astype(
            prepped_tensors[0].dtype
        )

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class SlerpMerge(MergeMethod):
    def name(self) -> str:
        return "slerp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "SLERP"

    @override
    def reference_url(self):
        return "https://en.wikipedia.org/wiki/Slerp"

    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="t", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        base_model: Optional[ModelReference],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **_kwargs,
    ) -> Task:
        return SlerpTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            t=parameters["t"],
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )


def lerp(
    t: float, v0: mindspore.Tensor, v1: mindspore.Tensor
) -> mindspore.Tensor:
    return (1 - t) * v0 + t * v1


def slerp(
    t: float,
    v0: mindspore.Tensor,
    v1: mindspore.Tensor,
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
) -> mindspore.Tensor:
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (mindspore.Tensor): Starting vector
        v1 (mindspore.Tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (mindspore.Tensor): Interpolation vector between v0 and v1
    """
    v0_f = v0.astype(mindspore.float32)
    v1_f = v1.astype(mindspore.float32)

    v0_copy = v0_f.copy()
    v1_copy = v1_f.copy()

    v0_norm = normalize(v0_f, eps)
    v1_norm = normalize(v1_f, eps)

    dot = (v0_norm * v1_norm).sum()

    if ops.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)

    theta_0 = ops.acos(dot)
    sin_theta_0 = ops.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = ops.sin(theta_t)

    s0 = ops.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return res


def normalize(v: mindspore.Tensor, eps: float) -> mindspore.Tensor:
    norm_v = ops.norm(v.astype(mindspore.float32))
    if norm_v > eps:
        v = v / norm_v
    return v
