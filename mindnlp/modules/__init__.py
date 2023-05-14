# Copyright 2022 Huawei Technologies Co., Ltd
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
"""modules init"""

from mindnlp.utils import less_min_pynative_first
from mindnlp._legacy.nn import transformer
from mindnlp.modules import encoder, decoder, embeddings, loss, attentions, crf, rnns, \
    accumulator
from mindnlp.modules.attentions import ScaledDotAttention, SelfAttention, \
    BinaryAttention, AdditiveAttention, CosineAttention, LocationAwareAttention, \
    LinearAttention
from mindnlp.modules.encoder import RNNEncoder, CNNEncoder
from mindnlp.modules.decoder import RNNDecoder
from mindnlp.modules.embeddings import Fasttext, Glove
from mindnlp.modules.crf import CRF
from mindnlp.modules.loss import RDropLoss, CMRC2018Loss
from mindnlp.modules.rnns import *
from mindnlp.modules.generation import *
from mindnlp.modules.accumulator import *

if less_min_pynative_first:
    from mindnlp._legacy.nn.transformer import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention
else:
    from mindspore.nn import Transformer, TransformerDecoder, TransformerEncoder, \
        TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention

__all__ = []

__all__.extend(transformer.__all__)
__all__.extend(encoder.__all__)
__all__.extend(decoder.__all__)
__all__.extend(embeddings.__all__)
__all__.extend(attentions.__all__)
__all__.extend(crf.__all__)
__all__.extend(loss.__all__)
__all__.extend(rnns.__all__)
__all__.extend(accumulator.__all__)
