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
"""
Callbacks.
"""
from mindnlp.metrics import accuracy, bleu, confusion_matrix, distinct, em_score, \
    f1, matthews, pearson, perplexity, precision, recall, rouge, spearman

from .perplexity import *
from .bleu import *
from .rouge import *
from .distinct import *
from .accuracy import *
from .precision import *
from .recall import *
from .f1 import *
from .matthews import *
from .pearson import *
from .spearman import *
from .em_score import *
from .confusion_matrix import *

__all__ = []
__all__.extend(accuracy.__all__)
__all__.extend(bleu.__all__)
__all__.extend(confusion_matrix.__all__)
__all__.extend(distinct.__all__)
__all__.extend(em_score.__all__)
__all__.extend(f1.__all__)
__all__.extend(matthews.__all__)
__all__.extend(pearson.__all__)
__all__.extend(perplexity.__all__)
__all__.extend(precision.__all__)
__all__.extend(recall.__all__)
__all__.extend(rouge.__all__)
__all__.extend(spearman.__all__)
