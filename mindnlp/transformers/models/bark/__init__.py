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
Encodec Model init
"""

from . import generation_configuration_bark, modeling_bark, configuration_bark, processing_bark
from .configuration_bark import *
from .generation_configuration_bark import *
from .modeling_bark import *
from .processing_bark import *

__all__ = []
__all__.extend(modeling_bark.__all__)
__all__.extend(processing_bark.__all__)
__all__.extend(configuration_bark.__all__)
__all__.extend(generation_configuration_bark.__all__)
