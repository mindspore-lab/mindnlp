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
Falcon Model.
"""

from . import configuration_gemma, modeling_gemma, tokenization_gemma, tokenization_gemma_fast

from .configuration_gemma import *
from .modeling_gemma import *
from .tokenization_gemma import *
from .tokenization_gemma_fast import *

__all__ = []
__all__.extend(configuration_gemma.__all__)
__all__.extend(modeling_gemma.__all__)
__all__.extend(tokenization_gemma.__all__)
__all__.extend(tokenization_gemma_fast.__all__)
