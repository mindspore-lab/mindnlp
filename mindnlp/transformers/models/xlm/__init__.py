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
XLM Models init
"""
from . import configuration_xlm, modeling_xlm, tokenization_xlm

from .configuration_xlm import *
from .modeling_xlm import *
from .tokenization_xlm import *

__all__ = []
__all__.extend(modeling_xlm.__all__)
__all__.extend(configuration_xlm.__all__)
__all__.extend(tokenization_xlm.__all__)
