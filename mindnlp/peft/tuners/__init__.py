# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Tuners"""

from .lora import LoraConfig, LoraModel
from .ia3 import IA3Config, IA3Model
from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
from .adalora import AdaLoraConfig, AdaLoraModel
from .lokr import LoKrConfig, LoKrModel
from .loha import LoHaConfig, LoHaModel
from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
from .prefix_tuning import PrefixTuningConfig, PrefixEncoder
from .prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from .multitask_prompt_tuning import MultitaskPromptEmbedding, MultitaskPromptTuningConfig, MultitaskPromptTuningInit
from .poly import PolyConfig, PolyModel
from .ln_tuning import LNTuningConfig, LNTuningModel
