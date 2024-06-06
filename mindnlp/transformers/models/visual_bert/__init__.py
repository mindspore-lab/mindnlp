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
"""VisualBERT model."""
from . import configuration_visual_bert, modeling_visual_bert

from .configuration_visual_bert import *
from .modeling_visual_bert import *

__all__ = []
__all__.extend(configuration_visual_bert.__all__)
__all__.extend(modeling_visual_bert.__all__)
