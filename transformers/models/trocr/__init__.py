# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""TrOCR Model init."""

from . import configuration_trocr, modeling_trocr, processing_trocr
from .configuration_trocr import *
from .modeling_trocr import *
from .processing_trocr import *

__all__ = []
__all__.extend(configuration_trocr.__all__)
__all__.extend(modeling_trocr.__all__)
__all__.extend(processing_trocr.__all__)
