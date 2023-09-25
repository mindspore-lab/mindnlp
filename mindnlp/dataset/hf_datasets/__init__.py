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
Directly load huggingface datasets
"""

from .hf_imdb import HF_IMDB, HF_IMDB_Process
from .hf_glue import HF_GLUE, HF_GLUE_Process
from .hf_msra_ner import HF_Msra_ner, HF_Msra_ner_Process
from .hf_ptb_text_only import HF_Ptb_text_only, HF_Ptb_text_only_Process
from .hf_cmrc2018 import HF_CMRC2018, HF_CMRC2018_Process
from .hf_duconv import hf_duconv, hf_duconv_process
from .hf_squad2 import HF_SQuAD2, HF_SQuAD2_Process
from .hf_squad import HF_SQuAD, HF_SQuAD_Process
from .hf_dureader_robust import HF_dureader_robust, HF_dureader_robust_Process
from .mt_eng_vietnamese import hf_mt_eng_vietnamese
from .hf_xfund import HF_XFUND
from .hf_funsd import HF_FUNSD
