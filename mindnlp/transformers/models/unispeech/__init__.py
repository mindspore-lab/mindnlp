# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Unispeech model
"""
from . import modeling_unispeech, configuration_unispeech
from .modeling_unispeech import *
from .configuration_unispeech import *

__all__ = []
__all__.extend(modeling_unispeech.__all__)
__all__.extend(configuration_unispeech.__all__)
