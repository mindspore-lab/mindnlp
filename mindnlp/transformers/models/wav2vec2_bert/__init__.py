# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Wav2Vec2-BERT Model"""
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, is_mindspore_available

from . import configuration_wav2vec2_bert
from . import modeling_wav2vec2_bert
from . import processing_wav2vec2_bert

from .configuration_wav2vec2_bert import *
from .modeling_wav2vec2_bert import *
from .processing_wav2vec2_bert import *

__all__ = []
__all__.extend(configuration_wav2vec2_bert.__all__)
__all__.extend(modeling_wav2vec2_bert.__all__)
__all__.extend(processing_wav2vec2_bert.__all__)

_import_structure = {
    "configuration_wav2vec2_bert": ["Wav2Vec2BertConfig"],
    "processing_wav2vec2_bert": ["Wav2Vec2BertProcessor"],
}


try:
    if not is_mindspore_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_wav2vec2_bert"] = [
        "Wav2Vec2BertForAudioFrameClassification",
        "Wav2Vec2BertForCTC",
        "Wav2Vec2BertForSequenceClassification",
        "Wav2Vec2BertForXVector",
        "Wav2Vec2BertModel",
        "Wav2Vec2BertPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_wav2vec2_bert import (
        Wav2Vec2BertConfig,
    )
    from .processing_wav2vec2_bert import Wav2Vec2BertProcessor

    try:
        if not is_mindspore_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_wav2vec2_bert import (
            Wav2Vec2BertForAudioFrameClassification,
            Wav2Vec2BertForCTC,
            Wav2Vec2BertForSequenceClassification,
            Wav2Vec2BertForXVector,
            Wav2Vec2BertModel,
            Wav2Vec2BertPreTrainedModel,
        )
