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
from .hf_glue import HF_GLUE
from .hf_msra_ner import HF_Msra_ner, HF_Msra_ner_Process
from .hf_ptb_text_only import HF_Ptb_text_only, HF_Ptb_text_only_Process
from .hf_squad2 import HF_SQuAD2
