# Copyright 2020 The HuggingFace Team. All rights reserved.
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
""" DETR model """

from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, is_mindspore_available, is_vision_available

from . import configuration_detr, feature_extraction_detr, image_processing_detr, modeling_detr

from .configuration_detr import *
from .feature_extraction_detr import *
from .image_processing_detr import *
from .modeling_detr import *

_import_structure = {"configuration_detr": ["DetrConfig", "DetrOnnxConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_detr"] = ["DetrFeatureExtractor"]
    _import_structure["image_processing_detr"] = ["DetrImageProcessor"]

try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_detr"] = [
        "DetrForObjectDetection",
        "DetrForSegmentation",
        "DetrModel",
        "DetrPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_detr import DetrConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_detr import DetrFeatureExtractor
        from .image_processing_detr import DetrImageProcessor

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_detr import (
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )

__all__ = []
__all__.extend(configuration_detr.__all__)
__all__.extend(feature_extraction_detr.__all__)
__all__.extend(image_processing_detr.__all__)
__all__.extend(modeling_detr.__all__)
