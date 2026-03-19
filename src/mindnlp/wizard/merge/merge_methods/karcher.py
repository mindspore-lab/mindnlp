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
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class KarcherTask(Task[mindspore.Tensor]):
    """
    Task for merging model weights using the Riemannian (Karcher) mean algorithm.

    The Karcher mean provides a geometrically meaningful way to average points on a manifold,
    which is particularly useful for merging model weights that can be interpreted as points
    on a hypersphere.
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    max_iter: int
    tol: float
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, mindspore.Tensor]) -> mindspore.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        model_tensors = list(tensors.values())
        threshold = self.max_tensor_mem_gb
        if (
            threshold is not None
            and self.split_pieces > 1
            and model_tensors
            and model_tensors[0].ndim >= 1
            and int(model_tensors[0].nbytes) > int(float(threshold) * (1024**3))
            and int(model_tensors[0].shape[0]) >= self.split_pieces
        ):
            total = int(model_tensors[0].shape[0])
            outputs = []
            for piece_idx in range(self.split_pieces):
                start = (total * piece_idx) // self.split_pieces
                end = (total * (piece_idx + 1)) // self.split_pieces
                if end <= start:
                    continue
                outputs.append(self._merge_core([t[start:end] for t in model_tensors]))
            if outputs:
                return ops.concat(outputs, axis=0)
        return self._merge_core(model_tensors)

    def _merge_core(self, model_tensors: List[mindspore.Tensor]) -> mindspore.Tensor:

        for i in range(1, len(model_tensors)):
            rectify_embed_sizes(self.weight_info, [model_tensors[0], model_tensors[i]])

        alphas = [1.0 / len(model_tensors)] * len(model_tensors)

        return karcher_merge_tensors(model_tensors, alphas, max_iter=self.max_iter, tol=self.tol)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class KarcherMerge(MergeMethod):
    """
    Implementation of the Karcher mean merge method.

    This method merges model weights using the Riemannian (Karcher) mean concept,
    which provides a geometrically meaningful way to average points on a manifold.
    """

    def name(self) -> str:
        return "karcher"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Karcher Mean"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://en.wikipedia.org/wiki/Karcher_mean"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="max_iter", required=False, default_value=10),
            ConfigParameterDef(name="tol", required=False, default_value=1e-5),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **_kwargs,
    ) -> Task:
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 10
        tol = parameters["tol"] if "tol" in parameters else 1e-5

        return KarcherTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            max_iter=max_iter,
            tol=tol,
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )


def karcher_merge_tensors(tensors, alphas, max_iter=10, tol=1e-5):
    """
    Implements weight fusion based on the Riemannian (Karcher) mean concept.

    Args:
        tensors: List of tensors to merge
        alphas: List of weights for each tensor
        max_iter: Maximum number of iterations for the Karcher mean algorithm
        tol: Convergence tolerance

    Returns:
        Merged tensor using Karcher mean algorithm
    """
    if len(tensors) == 1:
        return tensors[0]
    out_dtype = tensors[0].dtype
    work_dtype = choose_work_dtype(out_dtype)
    tensors = [t.astype(work_dtype) for t in tensors]

    norms = []
    units = []
    for t in tensors:
        t_float = t.astype(mindspore.float32)
        n = ops.norm(t_float)
        n_val = n.asnumpy().item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(ops.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).astype(work_dtype))

    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return ops.zeros_like(tensors[0])

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    u = ops.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u = u + a * ui
    norm_u = ops.norm(u.astype(mindspore.float32)).asnumpy().item()
    if norm_u < tol:
        u = valid_units[0].copy()
    else:
        u = (u / norm_u).astype(u.dtype)

    for _ in range(max_iter):
        T = ops.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            dot = ops.clamp(
                (
                    u.flatten().astype(mindspore.float32)
                    * ui.flatten().astype(mindspore.float32)
                ).sum(),
                -1.0,
                1.0,
            )
            theta = ops.acos(dot)
            theta_val = theta.asnumpy().item()
            if theta_val < tol:
                continue
            sin_theta = ops.sin(theta)
            T = T + a * (theta / sin_theta) * (ui - dot * u)

        norm_T = ops.norm(T.astype(mindspore.float32))
        if norm_T.asnumpy().item() < tol:
            break

        cos_norm_T = ops.cos(norm_T)
        sin_norm_T = ops.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).astype(u.dtype)

        u_norm = ops.norm(u.astype(mindspore.float32))
        if u_norm.asnumpy().item() > tol:
            u = (u / u_norm).astype(u.dtype)

    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n

    return (s * u).astype(out_dtype)
