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
'''Mask2Former Model.'''
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, is_mindspore_available, is_vision_available

from . import configuration_mask2former
from . import image_processing_mask2former
from . import modeling_mask2former

from .configuration_mask2former import *
from .image_processing_mask2former import *
from .modeling_mask2former import *

__all__ = []
__all__.extend(configuration_mask2former.__all__)
__all__.extend(image_processing_mask2former.__all__)
__all__.extend(modeling_mask2former.__all__)

_import_structure = {
    "configuration_mask2former": ["Mask2FormerConfig"],
}

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_mask2former"] = ["Mask2FormerImageProcessor"]

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mask2former"] = [
        "Mask2FormerForUniversalSegmentation",
        "Mask2FormerModel",
        "Mask2FormerPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_mask2former import Mask2FormerConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_mask2former import Mask2FormerImageProcessor

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mask2former import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
