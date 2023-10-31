# Copyright 2021 Huawei Technologies Co., Ltd
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
Models init
"""
from . import bart, bert, bloom, clip, codegen, cpm, ernie, glm, gpt, gpt_neo, gpt2, \
    llama, longformer, luke, megatron_bert, mobilebert, nezha, opt, pangu, roberta, rwkv, \
    t5, tinybert, xlm, xlm_roberta, gpt_bigcode
from .bart import *
from .bert import *
from .bloom import *
from .clip import *
from .codegen import *
from .cpm import *
from .ernie import *
from .glm import *
from .gpt import *
from .gpt_neo import *
from .gpt2 import *
from .llama import *
from .longformer import *
from .luke import *
from .megatron_bert import *
from .mobilebert import *
from .nezha import *
from .opt import *
from .pangu import *
from .roberta import *
from .rwkv import *
from .t5 import *
from .tinybert import *
from .xlm import *
from .xlm_roberta import *
from .gpt_bigcode import *

__all__ = []
__all__.extend(bart.__all__)
__all__.extend(bert.__all__)
__all__.extend(bloom.__all__)
__all__.extend(clip.__all__)
__all__.extend(codegen.__all__)
__all__.extend(cpm.__all__)
__all__.extend(ernie.__all__)
__all__.extend(glm.__all__)
__all__.extend(gpt.__all__)
__all__.extend(gpt_neo.__all__)
__all__.extend(gpt2.__all__)
__all__.extend(llama.__all__)
__all__.extend(longformer.__all__)
__all__.extend(luke.__all__)
__all__.extend(megatron_bert.__all__)
__all__.extend(mobilebert.__all__)
__all__.extend(nezha.__all__)
__all__.extend(opt.__all__)
__all__.extend(pangu.__all__)
__all__.extend(roberta.__all__)
__all__.extend(rwkv.__all__)
__all__.extend(t5.__all__)
__all__.extend(tinybert.__all__)
__all__.extend(xlm.__all__)
__all__.extend(xlm_roberta.__all__)
__all__.extend(gpt_bigcode.__all__)
