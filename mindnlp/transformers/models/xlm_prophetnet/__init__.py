# Copyright 2020 The HuggingFace Team. All rights reserved.
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
XLM-prophetnet Models init
"""
from . import configuration_xlm_prophetnet, modeling_xlm_prophetnet, tokenization_xlm_prophetnet

from .configuration_xlm_prophetnet import *
from .modeling_xlm_prophetnet import *
from .tokenization_xlm_prophetnet import *

__all__ = []
__all__.extend(modeling_xlm_prophetnet.__all__)
__all__.extend(configuration_xlm_prophetnet.__all__)
__all__.extend(tokenization_xlm_prophetnet.__all__)
