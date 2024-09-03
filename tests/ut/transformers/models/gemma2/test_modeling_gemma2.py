# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma2 model."""

import unittest

from pytest import mark

from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config
from mindnlp.utils import is_mindspore_available
from mindnlp.utils.testing_utils import (
    require_mindspore,
    require_mindspore_gpu,
    slow,
)

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


if is_mindspore_available():
    import mindspore
    from mindnlp.core import ops
    from mindnlp.transformers import (
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
    )


class Gemma2ModelTester(GemmaModelTester):
    if is_mindspore_available():
        config_class = Gemma2Config
        model_class = Gemma2Model
        for_causal_lm_class = Gemma2ForCausalLM
        for_sequence_class = Gemma2ForSequenceClassification
        for_token_class = Gemma2ForTokenClassification


@require_mindspore
class Gemma2ModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2ForTokenClassification)
    )
    all_generative_model_classes = ()

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    _torch_compile_test_ckpt = "google/gemma-2-9b"

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma2Config, hidden_size=37)

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Gemma2's outputs are expected to be different")
    def test_eager_matches_sdpa_inference(self):
        pass




