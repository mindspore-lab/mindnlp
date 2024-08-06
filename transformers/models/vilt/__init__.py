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
"""VilT model."""
from . import configuration_vilt, feature_extraction_vilt, image_processing_vilt, modeling_vilt, processing_vilt

from .processing_vilt import *
from .modeling_vilt import *
from .image_processing_vilt import *
from .feature_extraction_vilt import *
from .configuration_vilt import *

__all__ = []
__all__.extend(processing_vilt.__all__)
__all__.extend(modeling_vilt.__all__)
__all__.extend(image_processing_vilt.__all__)
__all__.extend(feature_extraction_vilt.__all__)
__all__.extend(configuration_vilt.__all__)
