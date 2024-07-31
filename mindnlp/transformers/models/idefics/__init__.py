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
"""Idefics model."""
from . import configuration_idefics, image_processing_idefics, modeling_idefics, perceiver, processing_idefics, vision
from .configuration_idefics import *
from .image_processing_idefics import *
from .modeling_idefics import *
from .perceiver import *
from .processing_idefics import *
from .vision import *

__all__ = []
__all__.extend(configuration_idefics.__all__)
__all__.extend(image_processing_idefics.__all__)
__all__.extend(modeling_idefics.__all__)
__all__.extend(perceiver.__all__)
__all__.extend(processing_idefics.__all__)
__all__.extend(vision.__all__)
