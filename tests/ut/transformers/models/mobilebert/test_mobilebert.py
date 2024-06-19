# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
# pylint: disable=R0904
"""
Test MobileBert
"""
import gc
import os
import unittest

import mindspore
from mindspore import Tensor
import numpy as np

from mindnlp.transformers.models.mobilebert.mobilebert import MobileBertSelfAttention, MobileBertSelfOutput,\
    MobileBertAttention, MobileBertIntermediate, OutputBottleneck, MobileBertOutput,\
    BottleneckLayer, Bottleneck, FFNOutput, FFNLayer, MobileBertLayer, MobileBertEncoder,\
    MobileBertEmbeddings, MobileBertPooler, MobileBertPredictionHeadTransform, \
    MobileBertLMPredictionHead, MobileBertModel, MobileBertForPreTraining, MobileBertForMaskedLM,\
    MobileBertOnlyNSPHead, MobileBertForNextSentencePrediction, MobileBertForSequenceClassification, \
    MobileBertForQuestionAnswering, MobileBertForMultipleChoice, MobileBertForTokenClassification,\
    MobileBertConfig
from .....common import MindNLPTestCase

class TestMobileBert(MindNLPTestCase):
    """
    Test TinyBert Models
    """

    def setUp(self):
        """
        Set up config
        """

        self.config = MobileBertConfig(num_hidden_layers=2)

    def test_mobilebert_embedding(self):
        """
        Test MobileBertEmbeddings
        """
        model = MobileBertEmbeddings(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 128)))

        output = model(input_ids)
        assert output.shape == (2, 128, 512)

    def test_mobilebert_selfattention(self):
        """
        Test MobileBertEmbeddings
        """
        model = MobileBertSelfAttention(self.config)
        query_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 128)), mindspore.float32)
        key_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 128)), mindspore.float32)
        value_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 512)), mindspore.float32)

        output = model(query_tensor, key_tensor, value_tensor)
        assert output[0].shape == (2, 8, 128)

    def test_mobilebert_selfoutput(self):
        """
        Test MobileBertSelfOutput
        """
        model = MobileBertSelfOutput(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (2, 128)), mindspore.float32)
        residual_tensor = Tensor(np.random.randint(0, 1000, (2, 128)))

        output = model(hidden_states, residual_tensor)
        assert output.shape == (2, 128)


    def test_mobilebert_attention(self):
        """
        Test MobileBertAttention
        """
        model = MobileBertAttention(self.config)
        query_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 128)), mindspore.float32)
        key_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 128)), mindspore.float32)
        value_tensor = Tensor(np.random.randint(0, 1000, (2, 8, 512)), mindspore.float32)
        layer_input = Tensor(np.random.randint(0, 1000, (2, 8, 128)), mindspore.float32)

        output = model(query_tensor, key_tensor, value_tensor, layer_input)
        assert output[0].shape == (2, 8, 128)

    def test_mobilebert_intermediate(self):
        """
        Test MobileBertIntermediate
        """
        model = MobileBertIntermediate(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (512, 128)), mindspore.float32)

        output = model(hidden_states)
        assert output.shape == (512, 512)

    def test_mobilebert_outputbottleneck(self):
        """
        Test OutputBottleneck
        """
        model = OutputBottleneck(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (512, 128)), mindspore.float32)
        residual_tensor = Tensor(np.random.randint(0, 1000, (512, 512)), mindspore.float32)

        output = model(hidden_states, residual_tensor)
        assert output.shape == (512, 512)

    def test_mobilebert_mobilebertoutput(self):
        """
        Test MobileBertOutput
        """
        model = MobileBertOutput(self.config)
        intermediate_states = Tensor(np.random.randint(0, 1000, (128, 512)), mindspore.float32)
        residual_tensor_1 = Tensor(np.random.randint(0, 1000, (128, 128)), mindspore.float32)
        residual_tensor_2 = Tensor(np.random.randint(0, 1000, (128, 512)), mindspore.float32)

        output = model(intermediate_states, residual_tensor_1, residual_tensor_2)
        assert output.shape == (128, 512)

    def test_mobilebert_bottlenecklayer(self):
        """
        Test BottleneckLayer
        """
        model = BottleneckLayer(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (128, 512)), mindspore.float32)

        output = model(hidden_states)
        assert output.shape == (128, 128)

    def test_mobilebert_bottleneck(self):
        """
        Test Bottleneck
        """
        model = Bottleneck(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (128, 512)), mindspore.float32)

        output = model(hidden_states)
        assert output[0].shape == (128, 128)
        assert output[1].shape == (128, 128)
        assert output[2].shape == (128, 512)
        assert output[3].shape == (128, 128)

    def test_mobilebert_ffnoutput(self):
        """
        Test FFNOutput
        """
        model = FFNOutput(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (128, 512)), mindspore.float32)
        residual_tensor = Tensor(np.random.randint(0, 1000, (128, 128)), mindspore.float32)

        output = model(hidden_states, residual_tensor)
        assert output.shape == (128, 128)

    def test_mobilebert_ffnlayer(self):
        """
        Test FFNLayer
        """
        model = FFNLayer(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (512, 128)), mindspore.float32)

        output = model(hidden_states)
        assert output.shape == (512, 128)

    def test_mobilebert_mobilebertlayer(self):
        """
        Test MobileBertLayer
        """
        model = MobileBertLayer(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (2, 128, 512)), mindspore.float32)

        output = model(hidden_states)
        assert output[0].shape == (2, 128, 512)
        assert output[1].shape == ()
        for i in range(2, 12):
            if i in (4, 7):
                assert output[i].shape == (2, 128, 512)
            else:
                assert output[i].shape == (2, 128, 128)

    def test_mobilebert_mobilebertencoder(self):
        """
        Test MobileBertEncoder
        """
        model = MobileBertEncoder(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (2, 128, 512)), mindspore.float32)
        head_mask = Tensor(np.random.randint(0, 1000, (128, 128, 128)), mindspore.float32)

        output = model(hidden_states, head_mask=head_mask)
        assert output[0].shape == (2, 128, 512)

    def test_mobilebert_mobilebertpooler(self):
        """
        Test MobileBertPooler
        """
        model = MobileBertPooler(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (512, 128, 512)), mindspore.float32)

        output = model(hidden_states)
        assert output.shape == (512, 512)

    def test_mobilebert_mobilebertpredictionheadtransform(self):
        """
        Test MobileBertPredictionHeadTransform
        """
        model = MobileBertPredictionHeadTransform(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (2, 512, 512)), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 512, 512)

    def test_mobilebert_mobilebertlmpredictionhead(self):
        """
        Test MobileBertLMPredictionHead
        """
        model = MobileBertLMPredictionHead(self.config)
        hidden_states = Tensor(np.random.randint(0, 1000, (2, 256, 512)), mindspore.float32)

        output = model(hidden_states)
        assert output.shape == (2, 256, 30522)

    def test_mobilebert_model(self):
        """
        Test MobileBertModel
        """
        model = MobileBertModel(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs[0].shape == (2, 8, 512)
        assert outputs[1].shape == (2, 512)
        outputs = model(input_ids, return_dict=True)
        assert outputs[0].shape == (2, 8, 512)
        assert outputs[1].shape == (2, 512)

    def test_mobilebert_for_pretraining(self):
        """
        Test MobileBertForPreTraining
        """
        model = MobileBertForPreTraining(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.prediction_logits.shape == (2, 8, 30522)
        assert outputs.seq_relationship_logits.shape == (2, 2)

    def test_mobilebert_for_maskedlm(self):
        """
        Test MobileBertForMaskedLM
        """
        model = MobileBertForMaskedLM(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 8, 30522)

    def test_mobilebert_onlynsphead(self):
        """
        Test MobileBertOnlyNSPHead
        """
        model = MobileBertOnlyNSPHead(self.config)
        pooled_output = Tensor(np.random.randint(0, 1000, (2, 512)), mindspore.float32)

        outputs = model(pooled_output)
        assert outputs.shape == (2, 2)

    def test_mobilebert_for_nextsentenceprediction(self):
        """
        Test MobileBertForNextSentencePrediction
        """
        model = MobileBertForNextSentencePrediction(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 2)

    def test_mobilebert_for_sequenceclassification(self):
        """
        Test MobileBertForSequenceClassification
        """
        model = MobileBertForSequenceClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 2)

    def test_mobilebert_for_questionanswering(self):
        """
        Test MobileBertForQuestionAnswering
        """
        model = MobileBertForQuestionAnswering(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.start_logits.shape == (2, 8)
        assert outputs.end_logits.shape == (2, 8)

    def test_mobilebert_for_multiplechoice(self):
        """
        Test MobileBertForMultipleChoice
        """
        model = MobileBertForMultipleChoice(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8, 256)), mindspore.int32)

        outputs = model(input_ids)
        assert outputs.logits.shape == (2, 8)

    def test_mobilebert_for_tokenclassification(self):
        """
        Test MobileBertForTokenClassification
        """
        model = MobileBertForTokenClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 1000, (2, 8)), mindspore.int32)

        outputs = model(input_ids)

        assert outputs.logits.shape == (2, 8, 2)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
