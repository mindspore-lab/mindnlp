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

"""ViT model."""
from . import configuration_videomae, modeling_videomae, feature_extraction_videomae, image_processing_videomae
from .modeling_videomae import *
from .configuration_videomae import *
from .image_processing_videomae import *
from .feature_extraction_videomae import *


__all__ = []
__all__.extend(modeling_videomae.__all__)
__all__.extend(configuration_videomae.__all__)
__all__.extend(image_processing_videomae.__all__)
__all__.extend(feature_extraction_videomae.__all__)
