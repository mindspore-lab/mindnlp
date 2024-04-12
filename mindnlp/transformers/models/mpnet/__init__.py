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
"""
MPNet Model.
"""
from . import configuration_mpnet, modeling_mpnet, tokenization_mpnet
from .configuration_mpnet import *
from .modeling_mpnet import *
from .tokenization_mpnet import *

__all__ = []
__all__.extend(configuration_mpnet.__all__)
__all__.extend(modeling_mpnet.__all__)
__all__.extend(tokenization_mpnet.__all__)
