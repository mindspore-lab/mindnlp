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
"""Parameter effcient fine tuning modules, like huggingface peft."""
from .mapping import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
    get_peft_config,
    get_peft_model
)

from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    # PeftModelForFeatureExtraction,
    # PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)

from .tuners import (
    LoraConfig,
    LoraModel,
)

from .utils import (
    # TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftType,
    TaskType,
    # bloom_model_postprocess_past_key_value,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    # prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    shift_tokens_right,
    # load_peft_weights,
)

from .config import (
    PeftConfig,
    PromptLearningConfig,
)
