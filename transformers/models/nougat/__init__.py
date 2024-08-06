# Copyright 2023 The HuggingFace Team. All rights reserved.
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
Nougat Model init
"""

from . import image_processing_nougat, processing_nougat, tokenization_nougat_fast
from .image_processing_nougat import *
from .processing_nougat import *
from .tokenization_nougat_fast import *


__all__ = []
__all__.extend(image_processing_nougat.__all__)
__all__.extend(processing_nougat.__all__)
__all__.extend(tokenization_nougat_fast.__all__)
