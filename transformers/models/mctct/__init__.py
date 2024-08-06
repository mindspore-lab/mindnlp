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
M-CTC-T Model.
"""

from . import configuration_mctct, feature_extraction_mctct, modeling_mctct, processing_mctct
from .configuration_mctct import *
from .feature_extraction_mctct import *
from .modeling_mctct import *
from .processing_mctct import *

__all__ = []
__all__.extend(configuration_mctct.__all__)
__all__.extend(feature_extraction_mctct.__all__)
__all__.extend(modeling_mctct.__all__)
__all__.extend(processing_mctct.__all__)
