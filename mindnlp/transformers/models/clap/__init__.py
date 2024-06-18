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
"""PyTorch clap model."""
from . import configuration_clap, processing_clap, modeling_clap, feature_extraction_clap
from .modeling_clap import (
            ClapModel,
            ClapPreTrainedModel,
            ClapTextModel,
            ClapTextModelWithProjection,
            ClapAudioModel,
            ClapAudioModelWithProjection,)
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
from .processing_clap import ClapProcessor
from .feature_extraction_clap import ClapFeatureExtractor

__all__ = []
__all__.extend(modeling_clap.__all__)
__all__.extend(configuration_clap.__all__)
__all__.extend(processing_clap.__all__)
__all__.extend(feature_extraction_clap.__all__)

_import_structure = {
    "configuration_clap": [
        "ClapAudioConfig",
        "ClapConfig",
        "ClapTextConfig",
    ],
    "processing_clap": ["ClapProcessor"],
    "modeling_clap": [
        "ClapModel",
        "ClapPreTrainedModel",
        "ClapTextModel",
        "ClapTextModelWithProjection",
        "ClapAudioModel",
        "ClapAudioModelWithProjection",
    ],
    "feature_extraction_clap": ["ClapFeatureExtractor"],
}
