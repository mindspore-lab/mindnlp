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
ChatGLM Model init
"""
from mindnlp.transformers.models.chatglm2.configuration_chatglm2 import ChatGLM2Config as ChatGLM3Config
from . import modeling_chatglm3, tokenization_chatglm3

from .modeling_chatglm3 import *
from .tokenization_chatglm3 import *

__all__ = ['ChatGLM3Config']
__all__.extend(modeling_chatglm3.__all__)
__all__.extend(tokenization_chatglm3.__all__)
