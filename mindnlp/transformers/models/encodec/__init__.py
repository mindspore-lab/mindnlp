# Copyright 2022 Huawei Technologies Co., Ltd
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
Ernie Model init
"""

from . import modeling_encodec, configuration_encodec, feature_extraction_encodec
from .configuration_encodec import *
from .modeling_encodec import *
from .feature_extraction_encodec import *

__all__ = []
__all__.extend(modeling_encodec.__all__)
__all__.extend(configuration_encodec.__all__)
__all__.extend(feature_extraction_encodec.__all__)
