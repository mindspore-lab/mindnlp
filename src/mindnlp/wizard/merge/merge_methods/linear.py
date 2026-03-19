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
from ..safe_ops import safe_stack
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class LinearMergeTask(Task[mindspore.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, mindspore.Tensor], **_kwargs
    ) -> mindspore.Tensor:
        keys = list(tensors.keys())
        tensor_list = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        threshold = self.max_tensor_mem_gb
        if (
            threshold is not None
            and self.split_pieces > 1
            and tensor_list
            and tensor_list[0].ndim >= 1
            and int(tensor_list[0].nbytes) > int(float(threshold) * (1024**3))
            and int(tensor_list[0].shape[0]) >= self.split_pieces
        ):
            total = int(tensor_list[0].shape[0])
            outputs = []
            for piece_idx in range(self.split_pieces):
                start = (total * piece_idx) // self.split_pieces
                end = (total * (piece_idx + 1)) // self.split_pieces
                if end <= start:
                    continue
                piece_tensors = [t[start:end] for t in tensor_list]
                outputs.append(self._merge(piece_tensors, weights))
            if outputs:
                return ops.concat(outputs, axis=0)

        return self._merge(tensor_list, weights)

    def _merge(
        self, tensors: List[mindspore.Tensor], weights: List[float]
    ) -> mindspore.Tensor:
        rectify_embed_sizes(self.weight_info, tensors)
        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )
        out_dtype = tensors[0].dtype
        work_dtype = choose_work_dtype(out_dtype)
        stacked = safe_stack(tensors, axis=0, out_dtype=work_dtype, op_name="linear.stack")
        weight_tensor = mindspore.Tensor(weights, dtype=work_dtype)
        while len(weight_tensor.shape) < len(stacked.shape):
            weight_tensor = weight_tensor.unsqueeze(-1)
        res = (weight_tensor * stacked).sum(axis=0)
        if self.normalize:
            res = res / weight_tensor.sum(axis=0)
        return res.astype(out_dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class LinearMerge(MergeMethod):
    def name(self) -> str:
        return "linear"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Linear"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2203.05482"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **_kwargs,
    ) -> Task:
        return LinearMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            weight_info=output_weight,
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )
