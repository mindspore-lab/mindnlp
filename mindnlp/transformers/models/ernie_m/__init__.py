# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
ErnieM Model init
"""

from . import configuration_ernie_m, modeling_ernie_m, tokenization_ernie_m, modeling_graph_ernie_m
from .configuration_ernie_m import *
from .modeling_ernie_m import *
from .modeling_graph_ernie_m import *
from .tokenization_ernie_m import *

__all__ = []
__all__.extend(configuration_ernie_m.__all__)
__all__.extend(modeling_ernie_m.__all__)
__all__.extend(modeling_graph_ernie_m.__all__)
__all__.extend(tokenization_ernie_m.__all__)
