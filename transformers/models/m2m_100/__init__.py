# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
M2m_100 Model.
"""
from . import configuration_m2m_100,modeling_m2m_100,tokenization_m2m_100
from .configuration_m2m_100 import *
from .modeling_m2m_100 import *
from .tokenization_m2m_100 import *
__all__ = []

__all__.extend(configuration_m2m_100.__all__)
__all__.extend(modeling_m2m_100.__all__)
__all__.extend(tokenization_m2m_100.__all__)
