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
MGP-STR Model.
"""
from . import (
    modeling_mgp_str,
    configuration_mgp_str,
    processing_mgp_str,
    tokenization_mgp_str,
)
from .modeling_mgp_str import *
from .configuration_mgp_str import *
from .processing_mgp_str import *
from .tokenization_mgp_str import *

__all__ = []
__all__.extend(modeling_mgp_str.__all__)
__all__.extend(configuration_mgp_str.__all__)
__all__.extend(processing_mgp_str.__all__)
__all__.extend(tokenization_mgp_str.__all__)
