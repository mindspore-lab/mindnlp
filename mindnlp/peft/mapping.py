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
# pylint: disable=C0201
"""mappings"""
from typing import Any, Dict

from mindspore import nn

from .config import PeftConfig
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
    LoraModel
)

MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    # "QUESTION_ANS": PeftModelForQuestionAnswering,
    # "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}


PEFT_TYPE_TO_CONFIG_MAPPING = {
    # "ADAPTION_PROMPT": AdaptionPromptConfig,
    # "PROMPT_TUNING": PromptTuningConfig,
    # "PREFIX_TUNING": PrefixTuningConfig,
    # "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    # "ADALORA": AdaLoraConfig,
    # "IA3": IA3Config,
}

PEFT_TYPE_TO_TUNER_MAPPING = {
    "LORA": LoraModel,
    # "ADALORA": AdaLoraModel,
    # "IA3": IA3Model,
}


def get_peft_config(config_dict: Dict[str, Any]):
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def get_peft_model(model: nn.Cell, peft_config: PeftConfig, adapter_name: str = "default") -> PeftModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    # no specific task_type and is not prompt_learning
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)

    # TODO: prompt learning
    # if peft_config.is_prompt_learning:
    #     # peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
