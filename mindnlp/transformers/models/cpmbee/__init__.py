# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
"""CPM Bee model"""
from . import configuration_cpmbee, tokenization_cpmbee, modeling_cpmbee
from .configuration_cpmbee import *
from .tokenization_cpmbee import *
from .modeling_cpmbee import *

__all__ = []
__all__.extend(configuration_cpmbee.__all__)
__all__.extend(tokenization_cpmbee.__all__)
__all__.extend(modeling_cpmbee.__all__)
