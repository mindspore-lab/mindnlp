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
Model output
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mindspore
from .generic import ModelOutput

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    r"""
    Base Model Output
    """
    last_hidden_state: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    r"""
    Causal LM Output
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    r"""
    QA Output
    """
    loss: Optional[mindspore.Tensor] = None
    start_logits: mindspore.Tensor = None
    end_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    r"""
    Sequence Classifier Output
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
