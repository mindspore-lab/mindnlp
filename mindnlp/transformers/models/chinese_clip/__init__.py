# Copyright 2022 Huawei Technologies Co., Ltd
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
Chinese CLIP Model
"""
from . import configuration_chinese_clip, image_processing_chinese_clip, modeling_chinese_clip, processing_chinese_clip

from .configuration_chinese_clip import *
from .image_processing_chinese_clip import *
from .modeling_chinese_clip import *
from .processing_chinese_clip import *

__all__ = []
__all__.extend(configuration_chinese_clip.__all__)
__all__.extend(image_processing_chinese_clip.__all__)
__all__.extend(modeling_chinese_clip.__all__)
__all__.extend(processing_chinese_clip.__all__)
