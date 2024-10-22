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
"""Vits Model Init"""
from . import configuration_vits, modeling_vits, tokenization_vits
from .modeling_vits import *
from .configuration_vits import *
from .tokenization_vits import *

__all__ = []
__all__.extend(modeling_vits.__all__)
__all__.extend(configuration_vits.__all__)
__all__.extend(tokenization_vits.__all__)
