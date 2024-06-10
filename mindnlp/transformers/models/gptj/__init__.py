# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GPT-J model."""
from . import modeling_gptj,configuration_gptj
from .modeling_gptj import (
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,)
from .configuration_gptj import GPTJConfig

__all__ = []
__all__.extend(modeling_gptj.__all__)
__all__.extend(configuration_gptj.__all__)

_import_structure = {"configuration_gptj": ["GPTJConfig"],
                     "modeling_gptj": ["GPTJForCausalLM",
        "GPTJForQuestionAnswering",
        "GPTJForSequenceClassification",
        "GPTJModel",
        "GPTJPreTrainedModel",
    ]}
