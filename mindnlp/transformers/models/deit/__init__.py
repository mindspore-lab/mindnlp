# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
DeiT Model.
"""
from . import configuration_deit, feature_extraction_deit, image_processing_deit, modeling_deit
from .configuration_deit import *
from .feature_extraction_deit import *
from .image_processing_deit import *
from .modeling_deit import *

__all__ = []
__all__.extend(configuration_deit.__all__)
__all__.extend(feature_extraction_deit.__all__)
__all__.extend(image_processing_deit.__all__)
__all__.extend(modeling_deit.__all__)
