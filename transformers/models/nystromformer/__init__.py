# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
model nystromformer init
"""
from typing import TYPE_CHECKING
from . import configuration_nystromformer, modeling_nystromformer
from .configuration_nystromformer import *
from .modeling_nystromformer import *

__all__ = []
__all__.extend(configuration_nystromformer.__all__)
__all__.extend(modeling_nystromformer.__all__)

_import_structure = {
    "configuration_nystromformer": ["NystromformerConfig"],
}

_import_structure["modeling_nystromformer"] = [
    "NystromformerForMaskedLM",
    "NystromformerForMultipleChoice",
    "NystromformerForQuestionAnswering",
    "NystromformerForSequenceClassification",
    "NystromformerForTokenClassification",
    "NystromformerLayer",
    "NystromformerModel",
    "NystromformerPreTrainedModel",
]
