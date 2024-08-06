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
'''OneFormer Model'''
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, is_mindspore_available, is_vision_available

from . import configuration_oneformer, image_processing_oneformer, modeling_oneformer, processing_oneformer

from .configuration_oneformer import *
from .image_processing_oneformer import *
from .modeling_oneformer import *
from .processing_oneformer import *

__all__ = []
__all__.extend(configuration_oneformer.__all__)
__all__.extend(image_processing_oneformer.__all__)
__all__.extend(modeling_oneformer.__all__)
__all__.extend(processing_oneformer.__all__)

_import_structure = {
    "configuration_oneformer": ["OneFormerConfig"],
    "processing_oneformer": ["OneFormerProcessor"],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_oneformer"] = ["OneFormerImageProcessor"]

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_oneformer"] = [
        "OneFormerForUniversalSegmentation",
        "OneFormerModel",
        "OneFormerPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_oneformer import OneFormerConfig
    from .processing_oneformer import OneFormerProcessor

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_oneformer import OneFormerImageProcessor
    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_oneformer import (
            OneFormerForUniversalSegmentation,
            OneFormerModel,
            OneFormerPreTrainedModel,
        )
