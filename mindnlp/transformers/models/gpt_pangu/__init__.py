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
PanGu_Alpha Model init
"""
from . import configuration_gptpangu, modeling_gptpangu, tokenization_gptpangu

from .configuration_gptpangu import *
from .modeling_gptpangu import *
from .tokenization_gptpangu import *

__all__ = []
__all__.extend(modeling_gptpangu.__all__)
__all__.extend(configuration_gptpangu.__all__)
__all__.extend(tokenization_gptpangu.__all__)
