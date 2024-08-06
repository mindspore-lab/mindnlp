# Copyright 2024 The HuggingFace Team. All rights reserved.
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
Deta Model.
"""
from . import (
    configuration_deta,
    image_processing_deta,
    modeling_deta,
)
from .configuration_deta import *
from .image_processing_deta import *
from .modeling_deta import *

__all__ = []
__all__.extend(configuration_deta.__all__)
__all__.extend(image_processing_deta.__all__)
__all__.extend(modeling_deta.__all__)
