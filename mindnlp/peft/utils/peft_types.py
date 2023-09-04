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
"""
Peft types and Task types.
"""
import enum

class PeftType(str, enum.Enum):
    """
    PeftType.
    """
    # PROMPT_TUNING = "PROMPT_TUNING"
    # P_TUNING = "P_TUNING"
    # PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    # ADALORA = "ADALORA"
    # ADAPTION_PROMPT = "ADAPTION_PROMPT"
    # IA3 = "IA3"


class TaskType(str, enum.Enum):
    """
    TaskType
    """
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
