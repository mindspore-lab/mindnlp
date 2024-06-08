"""Wav2Vec2-Conformer based on MindSpore"""

from typing import TYPE_CHECKING
from mindnlp.utils import is_mindspore_available
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
from . import configuration_wav2vec2_conformer, modeling_wav2vec2_conformer

if is_mindspore_available():
    from .modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerForAudioFrameClassification,
        Wav2Vec2ConformerForCTC,
        Wav2Vec2ConformerForPreTraining,
        Wav2Vec2ConformerForSequenceClassification,
        Wav2Vec2ConformerForXVector,
        Wav2Vec2ConformerModel,
        Wav2Vec2ConformerPreTrainedModel,
    )

__all__ = []
__all__.extend(configuration_wav2vec2_conformer.__all__)
__all__.extend(modeling_wav2vec2_conformer.__all__)
