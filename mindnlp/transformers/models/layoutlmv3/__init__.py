# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# ============================================================================
"""
LayoutLMv3 Model init
"""
from .import (configuration_layoutlmv3, feature_extraction_layoutlmv3, image_processing_layoutlmv3, modeling_layoutlmv3,
              processing_layoutlmv3, tokenization_layoutlmv3, tokenization_layoutlmv3_fast)

from .configuration_layoutlmv3 import *
from .feature_extraction_layoutlmv3 import *
from .image_processing_layoutlmv3 import *
from .modeling_layoutlmv3 import *
from .processing_layoutlmv3 import *
from .tokenization_layoutlmv3 import *
from .tokenization_layoutlmv3_fast import *

__all__ = []
__all__.extend(configuration_layoutlmv3.__all__)
__all__.extend(feature_extraction_layoutlmv3.__all__)
__all__.extend(image_processing_layoutlmv3.__all__)
__all__.extend(modeling_layoutlmv3.__all__)
__all__.extend(processing_layoutlmv3.__all__)
__all__.extend(tokenization_layoutlmv3.__all__)
__all__.extend(tokenization_layoutlmv3_fast.__all__)
