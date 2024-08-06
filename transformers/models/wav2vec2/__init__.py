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

''' Wav2Vec2 Model '''

from . import configuration_wav2vec2, feature_extraction_wav2vec2, processing_wav2vec2, tokenization_wav2vec2, modeling_wav2vec2
from .configuration_wav2vec2 import WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2Config
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .processing_wav2vec2 import Wav2Vec2Processor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer
from .modeling_wav2vec2 import (
    WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForMaskedLM,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

__all__ = []
__all__.extend(configuration_wav2vec2.__all__)
__all__.extend(feature_extraction_wav2vec2.__all__)
__all__.extend(processing_wav2vec2.__all__)
__all__.extend(tokenization_wav2vec2.__all__)
__all__.extend(modeling_wav2vec2.__all__)
