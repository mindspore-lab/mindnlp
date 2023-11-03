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
"""Test Ernie"""
import pytest
import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.transformers import (
    UIEConfig,
    UIE
)
from mindnlp.transformers.models.ernie.ernie import (
    ErnieForSequenceClassification,
    ErnieForQuestionAnswering,
    ErnieForTokenClassification,
    ErnieForMultipleChoice,
    ErnieLMPredictionHead,
    ErnieOnlyMLMHead,
    ErnieForMaskedLM,
    ErnieForPretraining,
    ErniePretrainingHeads
)

import mindnlp


class TestModelingErnie(unittest.TestCase):
    r"""
    Test Ernie
    """

    def setUp(self):
        """
        Set up.
        """
        super().setUp()
        self.config = UIEConfig(
            vocab_size=1000,
            num_hidden_layers=2,
            hidden_size=128,
            num_hidden_layers2=2,
            num_attention_heads=2,
            intermediate_size=256,
        )

    def test_ernie_model(self):
        """
        Test ErnieModel
        """

    def test_uie_model(self):
        """
        Test UIEModel
        """

        model = UIE(self.config)


        input_ids = Tensor(np.random.randint(0, 100, (1, 20)), dtype=mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 20)
        assert outputs[1].shape == (1, 20)

    def test_sequenceclassification_model(self):
        """
        Test ErnieForSequenceClassification
        """
        model = ErnieForSequenceClassification(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (1, 20)), dtype=mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs.shape == (1, 2)

    def test_questionanswering_model(self):
        """
        Test ErnieForSequenceClassification
        """
        model = ErnieForQuestionAnswering(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (1, 20)), dtype=mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 20)
        assert outputs[1].shape == (1, 20)

    def test_tokenclassification_model(self):
        """
        Test ErnieForSequenceClassification
        """
        model = ErnieForTokenClassification(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (1, 20)), dtype=mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs.shape == (1, 20, 2)

    def test_lmpredictionhead_model(self):
        """
        Test LMPredictionHead
        """
        model = ErnieLMPredictionHead(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (128, 128)), dtype=mindspore.float32)
        outputs = model(input_ids)
        assert outputs.shape == (128, 1000)

    def test_erniepretrainingheads_model(self):
        """
        Test ErniePretrainingHeads
        """

        model = ErniePretrainingHeads(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (128, 128)), dtype=mindspore.float32)
        outputs = model(input_ids,input_ids)
        assert outputs[0].shape == (128,1000)
        assert outputs[1].shape == (128,2)

    def test_ernieonlymlmhead_model(self):
        """
        Test ErnieOnlyMLMHead
        """

        model = ErnieOnlyMLMHead(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (128,128)), dtype=mindspore.float32)
        outputs = model(input_ids)
        assert outputs.shape == (128,1000)

    def test_ernieformaskedlm_model(self):
        """
        Test ErnieForMaskedLM
        """
        model = ErnieForMaskedLM(self.config)



        input_np = Tensor(np.random.randint(0,100, (1,200)),dtype=mindspore.int32)
        outputs = model(input_np)
        assert outputs.shape == (1,200,1000)

    def test_ernieformultiplechoice_model(self):
        """
        Test ErnieForMultipleChoice
        """
        model = ErnieForMultipleChoice(self.config)



        input_np = Tensor(np.random.randint(0,100, (10,10,20)),dtype=mindspore.int32)
        outputs = model(input_np)
        assert outputs.shape == (50,2)

    def test_ernie_for_pretraining_model(self):
        """
        Test ErnieForPretraining
        """
        model = ErnieForPretraining(self.config)



        input_ids = Tensor(np.random.randint(
            0, 100, (1,128)), dtype=mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1,128,1000)
        assert outputs[1].shape == (1,2)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = UIE.from_pretrained("uie-base")
