# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from typing import Dict, List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from typing_extensions import override

from ..architecture.base import WeightInfo
from ..common import ModelReference
from ..dtype_policy import choose_work_dtype
from ..graph import Task
from ..safe_ops import safe_abs
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class DynamicThresholdFusion:
    def approximate_quantiles(self, tensor, q):
        flat_tensor = tensor.reshape(-1)

        if flat_tensor.numel() > 1e6:
            perm = ops.randperm(flat_tensor.numel())[:1000000]
            flat_tensor = flat_tensor[perm]

        sorted_tensor, _ = ops.sort(flat_tensor)

        quantile_indices = (q * (sorted_tensor.numel() - 1)).astype(mindspore.int64)

        return sorted_tensor[quantile_indices]

    def calculate_dynamic_threshold(self, importance_scores):
        median = self.approximate_quantiles(
            importance_scores, mindspore.Tensor([0.5])
        )[0]
        q1, q3 = self.approximate_quantiles(
            importance_scores, mindspore.Tensor([0.25, 0.75])
        )

        iqr = q3 - q1

        dynamic_threshold = median + 1.5 * iqr

        return dynamic_threshold

    def compute_fusion_mask(self, importance_scores):
        threshold = self.calculate_dynamic_threshold(importance_scores)
        fusion_mask = (importance_scores >= threshold).astype(mindspore.float32)
        return fusion_mask, threshold


class ArceeFusionMergeTask(Task[mindspore.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
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
            raise RuntimeError("ArceeFusion merge expects exactly two models")
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
        out_dtype = prepped_tensors[0].dtype
        work_dtype = choose_work_dtype(out_dtype)
        rectify_embed_sizes(self.weight_info, prepped_tensors)
        prepped_tensors = [t.astype(work_dtype) for t in prepped_tensors]
        importance_scores = self._compute_importance(prepped_tensors[1], prepped_tensors[0])
        dynamic_threshold_fusion = DynamicThresholdFusion()
        fusion_mask, _threshold = dynamic_threshold_fusion.compute_fusion_mask(importance_scores)
        delta = prepped_tensors[1] - prepped_tensors[0]
        masked_delta = delta * fusion_mask
        fused = prepped_tensors[0] + masked_delta
        return fused.astype(out_dtype)

    def _compute_importance(
        self, params: mindspore.Tensor, base_params: mindspore.Tensor, eps: float = 1e-8
    ) -> mindspore.Tensor:
        diff = safe_abs(params - base_params, out_dtype=params.dtype, op_name="arcee.diff_abs")
        p = ops.softmax(params.astype(mindspore.float32), axis=-1) + eps
        q = ops.softmax(base_params.astype(mindspore.float32), axis=-1) + eps
        kl_div = (p * ops.log(p / q)).sum(axis=-1)
        return diff * kl_div.unsqueeze(-1)


class ArceeFusionMerge(MergeMethod):
    def name(self) -> str:
        return "arcee_fusion"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Arcee Fusion"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arcee.ai"

    def parameters(self) -> List[ConfigParameterDef]:
        return []

    def make_task(  # pylint: disable=too-many-positional-arguments
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **kwargs,
    ) -> Task[mindspore.Tensor]:
        return ArceeFusionMergeTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )
