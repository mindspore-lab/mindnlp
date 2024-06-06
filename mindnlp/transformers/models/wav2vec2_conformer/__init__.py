from typing import TYPE_CHECKING

from mindnlp.utils import is_mindspore_available

# 配置结构导入
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig

# 检查MindSpore是否可用
if is_mindspore_available():
    # 直接导入所有Wav2Vec2Conformer相关模型类
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