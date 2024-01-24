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
from . import albert, auto, bart, bert, bloom, clip, codegen, cpm, ernie, ernie_m, chatglm, chatglm2, gpt, gpt_neo, gpt2, \
    llama, longformer, luke, mbart, megatron_bert, mistral, mobilebert, nezha, opt, pangu, phi_2, roberta, rwkv, \
    seamless_m4t, seamless_m4t_v2, t5, tinybert, xlm, xlm_roberta, gpt_bigcode, whisper, bark, encodec, \
    graphormer
from .albert import *
from .auto import *
from .bark import *
from .bart import *
from .bert import *
from .bloom import *
from .clip import *
from .codegen import *
from .cpm import *
from .ernie import *
from .ernie_m import *
from .chatglm import *
from .chatglm2 import *
from .gpt import *
from .gpt_neo import *
from .gpt2 import *
from .graphormer import *
from .llama import *
from .longformer import *
from .luke import *
from .mbart import *
from .megatron_bert import *
from .mistral import *
from .mobilebert import *
from .nezha import *
from .opt import *
from .pangu import *
from .phi_2 import *
from .roberta import *
from .rwkv import *
from .t5 import *
from .seamless_m4t import *
from .seamless_m4t_v2 import *
from .tinybert import *
from .xlm import *
from .xlm_roberta import *
from .gpt_bigcode import *
from .whisper import *
from .encodec import *

__all__ = []
__all__.extend(albert.__all__)
__all__.extend(auto.__all__)
__all__.extend(bart.__all__)
__all__.extend(bert.__all__)
__all__.extend(bloom.__all__)
__all__.extend(clip.__all__)
__all__.extend(codegen.__all__)
__all__.extend(cpm.__all__)
__all__.extend(ernie.__all__)
__all__.extend(ernie_m.__all__)
__all__.extend(chatglm.__all__)
__all__.extend(chatglm2.__all__)
__all__.extend(gpt.__all__)
__all__.extend(gpt_neo.__all__)
__all__.extend(gpt2.__all__)
__all__.extend(graphormer.__all__)
__all__.extend(llama.__all__)
__all__.extend(longformer.__all__)
__all__.extend(luke.__all__)
__all__.extend(mbart.__all__)
__all__.extend(megatron_bert.__all__)
__all__.extend(mistral.__all__)
__all__.extend(mobilebert.__all__)
__all__.extend(nezha.__all__)
__all__.extend(opt.__all__)
__all__.extend(pangu.__all__)
__all__.extend(phi_2.__all__)
__all__.extend(roberta.__all__)
__all__.extend(rwkv.__all__)
__all__.extend(t5.__all__)
__all__.extend(seamless_m4t.__all__)
__all__.extend(seamless_m4t_v2.__all__)
__all__.extend(tinybert.__all__)
__all__.extend(xlm.__all__)
__all__.extend(xlm_roberta.__all__)
__all__.extend(gpt_bigcode.__all__)
__all__.extend(whisper.__all__)
__all__.extend(bark.__all__)
__all__.extend(encodec.__all__)
