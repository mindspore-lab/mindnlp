# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# ============================================================================
"""
LLAMA Model init
"""
from . import configuration_llama, modeling_llama, tokenization_llama, tokenization_llama_fast, \
    tokenization_code_llama, tokenization_code_llama_fast

from .modeling_llama import *
from .configuration_llama import *
from .tokenization_llama import *
from .tokenization_llama_fast import *
from .tokenization_code_llama import *
from .tokenization_code_llama_fast import *

__all__ = []
__all__.extend(modeling_llama.__all__)
__all__.extend(configuration_llama.__all__)
__all__.extend(tokenization_llama.__all__)
__all__.extend(tokenization_llama_fast.__all__)
__all__.extend(tokenization_code_llama.__all__)
__all__.extend(tokenization_code_llama_fast.__all__)
