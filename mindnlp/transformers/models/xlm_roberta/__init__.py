# Copyright 2023 Huawei Technologies Co., Ltd
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
XLM Roberta init
"""
from . import configuration_xlm_roberta, modeling_xlm_roberta, \
    tokenization_xlm_roberta, tokenization_xlm_roberta_fast

from .configuration_xlm_roberta import *
from .modeling_xlm_roberta import *
from .tokenization_xlm_roberta import *
from .tokenization_xlm_roberta_fast import *

__all__ = []
__all__.extend(configuration_xlm_roberta.__all__)
__all__.extend(modeling_xlm_roberta.__all__)
__all__.extend(tokenization_xlm_roberta.__all__)
__all__.extend(tokenization_xlm_roberta_fast.__all__)
