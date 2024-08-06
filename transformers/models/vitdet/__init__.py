# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Vitdet Model Init"""
from . import configuration_vitdet, modeling_vitdet
from .modeling_vitdet import *
from .configuration_vitdet import *

__all__ = []
__all__.extend(modeling_vitdet.__all__)
__all__.extend(configuration_vitdet.__all__)
