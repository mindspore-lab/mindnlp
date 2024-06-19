# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Swin Model."""
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, is_mindspore_available

from . import configuration_swin
from . import modeling_swin

from .configuration_swin import *
from .modeling_swin import *

__all__ = []
__all__.extend(configuration_swin.__all__)
__all__.extend(modeling_swin.__all__)

_import_structure = {"configuration_swin": ["SwinConfig"]}


try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_swin"] = [
        "SwinForImageClassification",
        "SwinForMaskedImageModeling",
        "SwinModel",
        "SwinPreTrainedModel",
        "SwinBackbone",
    ]


if TYPE_CHECKING:
    from .configuration_swin import SwinConfig

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swin import (
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )
