# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""Speech2Text2 model init"""
from . import configuration_speech_to_text_2, modeling_speech_to_text_2, tokenization_speech_to_text_2, \
    processing_speech_to_text_2
from .configuration_speech_to_text_2 import *
from .modeling_speech_to_text_2 import *
from .tokenization_speech_to_text_2 import *
from .processing_speech_to_text_2 import *

__all__ = []
__all__.extend(configuration_speech_to_text_2.__all__)
__all__.extend(modeling_speech_to_text_2.__all__)
__all__.extend(tokenization_speech_to_text_2.__all__)
__all__.extend(processing_speech_to_text_2.__all__)
