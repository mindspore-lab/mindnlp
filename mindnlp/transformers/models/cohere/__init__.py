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
Cohere Model init
"""
from . import configuration_cohere, modeling_cohere, tokenization_cohere_fast

from .configuration_cohere import *
from .modeling_cohere import *
from .tokenization_cohere_fast import *

__all__ = []
__all__.extend(configuration_cohere.__all__)
__all__.extend(modeling_cohere.__all__)
__all__.extend(tokenization_cohere_fast.__all__)
