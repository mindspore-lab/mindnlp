# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MusicGen Melody Model init
"""
from . import configuration_musicgen_melody, modeling_musicgen_melody, feature_extraction_musicgen_melody, processing_musicgen_melody
from .configuration_musicgen_melody import *
from .modeling_musicgen_melody import *
from .feature_extraction_musicgen_melody import *
from .processing_musicgen_melody import *

__all__ = []
__all__.extend(configuration_musicgen_melody.__all__)
__all__.extend(modeling_musicgen_melody.__all__)
__all__.extend(feature_extraction_musicgen_melody.__all__)
__all__.extend(processing_musicgen_melody.__all__)
