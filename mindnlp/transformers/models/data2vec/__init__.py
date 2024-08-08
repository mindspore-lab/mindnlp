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
data2vec Model init.
"""
from . import configuration_data2vec_audio,modeling_data2vec_audio
from .configuration_data2vec_audio import *
from .modeling_data2vec_audio import *
from . import configuration_data2vec_text, modeling_data2vec_text
from .configuration_data2vec_text import *
from .modeling_data2vec_text import *

__all__ = []

__all__.extend(configuration_data2vec_audio.__all__)
__all__.extend(modeling_data2vec_audio.__all__)
__all__.extend(configuration_data2vec_text.__all__)
__all__.extend(modeling_data2vec_text.__all__)
