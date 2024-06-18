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
"""Splinter model init."""
from . import (
    configuration_splinter,
    modeling_splinter,
    tokenization_splinter,
    tokenization_splinter_fast,
)
from .configuration_splinter import *
from .modeling_splinter import *
from .tokenization_splinter import *
from .tokenization_splinter_fast import *

__all__ = []
__all__.extend(configuration_splinter.__all__)
__all__.extend(modeling_splinter.__all__)
__all__.extend(tokenization_splinter.__all__)
__all__.extend(tokenization_splinter_fast.__all__)
