# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

import logging
from typing import Any, Dict, List, Optional

import mindspore  # pylint: disable=import-error
from mindspore import ops  # pylint: disable=import-error
from typing_extensions import override

from ..architecture.base import WeightInfo
from ..common import ImmutableMap, ModelReference
from ..dtype_policy import choose_work_dtype
from ..graph import Task
from ..safe_ops import safe_norm, safe_stack
from .base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from .rectify_embed import rectify_embed_sizes


class ModelStockMergeTask(Task[mindspore.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    filter_wise: bool = False
    split_pieces: int = 1
    max_tensor_mem_gb: Optional[float] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, mindspore.Tensor]) -> mindspore.Tensor:
        if len(tensors) == 1 and self.base_model in tensors:
            return tensors[self.base_model]
        if len(tensors) < 3:
            if self.weight_info.optional:
                logging.warning(
                    f"Optional weight {self.weight_info.name} not present in enough models, discarding"
                )
                return None

            raise ValueError(
                "ModelStockMerge requires at least 3 models (base plus two+ others)"
            )

        w_0, ws = self.get_rectified_weights(tensors)
        threshold = self.max_tensor_mem_gb
        if (
            threshold is not None
            and self.split_pieces > 1
            and self.filter_wise
            and w_0.ndim >= 1
            and int(w_0.nbytes) > int(float(threshold) * (1024**3))
            and int(w_0.shape[0]) >= self.split_pieces
        ):
            total = int(w_0.shape[0])
            outputs = []
            for piece_idx in range(self.split_pieces):
                start = (total * piece_idx) // self.split_pieces
                end = (total * (piece_idx + 1)) // self.split_pieces
                if end <= start:
                    continue
                outputs.append(
                    self._merge_core(
                        w_0[start:end],
                        [w[start:end] for w in ws],
                        filter_wise=True,
                    )
                )
            if outputs:
                return ops.concat(outputs, axis=0)

        return self._merge_core(w_0, ws, filter_wise=self.filter_wise)

    def _merge_core(
        self, w_0: mindspore.Tensor, ws: List[mindspore.Tensor], *, filter_wise: bool
    ) -> mindspore.Tensor:
        out_dtype = w_0.dtype
        work_dtype = choose_work_dtype(out_dtype)
        w_0 = w_0.astype(work_dtype)
        ws = [w.astype(work_dtype) for w in ws]
        out_shape = w_0.shape

        if filter_wise:
            if w_0.ndim == 1:
                w_0 = w_0.unsqueeze(0)
                ws = [w.unsqueeze(0) for w in ws]
        else:
            w_0 = w_0.reshape(-1)
            ws = [w.reshape(-1) for w in ws]

        offsets = [w - w_0 for w in ws]
        cos_thetas = []
        for i, w_0_offset in enumerate(offsets):
            for j in range(i + 1, len(offsets)):
                w_1_offset = offsets[j]
                norm_product = safe_norm(
                    w_0_offset,
                    axis=-1,
                    out_dtype=mindspore.float32,
                    op_name="model_stock.norm0",
                ) * safe_norm(
                    w_1_offset,
                    axis=-1,
                    out_dtype=mindspore.float32,
                    op_name="model_stock.norm1",
                )
                cos_theta = (
                    (w_0_offset * w_1_offset).sum(axis=-1) / norm_product.clamp(min=1e-6)
                ).clamp(-1, 1)
                cos_thetas.append(cos_theta)

        cos_theta = safe_stack(
            cos_thetas, out_dtype=work_dtype, op_name="model_stock.stack"
        ).mean(axis=0).unsqueeze(-1)
        N = len(ws)
        t = (N * cos_theta) / (1 + (N - 1) * cos_theta)
        w_avg = sum(ws) / len(ws)
        w_h = t * w_avg + (1 - t) * w_0
        return w_h.reshape(out_shape).astype(out_dtype)

    def get_rectified_weights(self, tensors: Dict[ModelReference, mindspore.Tensor]):
        if self.base_model not in tensors:
            raise ValueError("Base model tensor not found")

        all_weights = [tensors[self.base_model]] + [
            tensors[k] for k in tensors if k != self.base_model
        ]
        rectify_embed_sizes(self.weight_info, all_weights)
        w_0 = all_weights[0]
        ws = all_weights[1:]
        return w_0, ws

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class ModelStockMerge(MergeMethod):
    def name(self) -> str:
        return "model_stock"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Model Stock"

    @override
    def reference_url(self):
        return "https://arxiv.org/abs/2403.19522"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="filter_wise", required=False, default_value=False)
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        split_pieces: int = 1,
        max_tensor_mem_gb: Optional[float] = None,
        **_kwargs,
    ) -> Task:
        return ModelStockMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            filter_wise=parameters["filter_wise"],
            split_pieces=split_pieces,
            max_tensor_mem_gb=max_tensor_mem_gb,
        )
