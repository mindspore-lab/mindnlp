# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# Modified for MindSpore/Ascend NPU by MindNLP Wizard contributors.
#

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import mindspore  # pylint: disable=import-error
from pydantic import BaseModel
from typing_extensions import TypeAlias

from ..architecture.base import WeightInfo
from ..common import ImmutableMap, ModelReference
from ..graph import Task
from ..io.tasks import GatherTensors
from ..tokenizer import PermutedEmbeddings


class TensorDictWrapper(Task[Dict[ModelReference, mindspore.Tensor]]):
    tensors: ImmutableMap[ModelReference, Task[mindspore.Tensor]]

    def arguments(self) -> Dict[str, Task]:
        return {
            k.model_dump_json(
                exclude_none=True, exclude_defaults=True, round_trip=True
            ): v
            for k, v in self.tensors.items()
        }

    def execute(self, **kwargs) -> Dict[ModelReference, mindspore.Tensor]:
        return {ModelReference.model_validate_json(k): v for k, v in kwargs.items()}


MergeTensorInput: TypeAlias = Union[
    GatherTensors, PermutedEmbeddings, TensorDictWrapper
]


class ConfigParameterDef(BaseModel):
    name: str
    required: bool = False
    default_value: Any = None


class MergeMethod(ABC):
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return []

    def parameters(self) -> List[ConfigParameterDef]:
        return []

    @abstractmethod
    def name(self) -> str: ...

    def pretty_name(self) -> Optional[str]:
        return None

    def reference_url(self) -> Optional[str]:
        return None

    @abstractmethod
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task: ...
