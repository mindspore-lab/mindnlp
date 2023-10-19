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
Test XLM
"""
import gc
import os
import unittest
import pytest
import numpy as np
import mindspore
from mindnlp.transformers import XLMConfig, XLMModel, XLMForSequenceClassification, XLMForQuestionAnsweringSimple, \
    XLMForTokenClassification, XLMForQuestionAnswering, XLMForMultipleChoice
from mindnlp.transformers.models.xlm.xlm import XLMPredLayer, MultiHeadAttention, \
    TransformerFFN, XLMWithLMHeadModel


class TestXlm(unittest.TestCase):
    """
    Test XLM Models
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = XLMConfig(vocab_size=22,
                                emb_dim=24,
                                n_layers=2,
                                n_heads=8,
                                max_position_embeddings=128,
                                batch_size=1)

    def test_xlm_predlayer(self):
        """
        Test xlm_XLMPredLayer
        """
        xlm_predlayer = XLMPredLayer(self.config)
        input_ids = mindspore.Tensor(np.random.randint(
            0, 1000, (2, 24)), mindspore.float32)
        output = xlm_predlayer(input_ids)
        assert output[0].shape == (2, 22)

    def test_xlm_multiheadattention(self):
        """
        test xlm_multiheadattention
        """
        xlm_multiheadattention = MultiHeadAttention(n_heads=self.config.n_heads,
                                                        dim=self.config.emb_dim,
                                                        config=self.config)
        input_ids = mindspore.Tensor(np.random.randint(
            0, 1000, (1, 2, 24)), mindspore.float32)
        mask = mindspore.Tensor(np.random.randint(
            0, 1000, (1, 1, 1, 2)), mindspore.float32)
        output = xlm_multiheadattention(input=input_ids, mask=mask)
        assert output[0].shape == (1, 2, 24)

    def test_xlm_transformerffn(self):
        """
        test xlm_TransformerFFN
        """
        xlm_transformerffn = TransformerFFN(
            self.config.emb_dim//8, self.config.emb_dim//2, self.config.emb_dim, self.config)
        input_ids = mindspore.Tensor(np.random.randint(
            0, 1000, (12, 3)), mindspore.float32)
        output = xlm_transformerffn(input_ids)
        assert output.shape == (12, 24)

    def test_xlm_xlmwithlmheadmodel(self):
        """
        test xlm_xlmwithLmheadmodel
        """
        xlm_xlmwithlmheadmodel = XLMWithLMHeadModel(self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 22, (4, 8)), mindspore.int32)
        output = xlm_xlmwithlmheadmodel(input_ids=input_ids1)
        assert output[0].shape == (4, 8, 22)

    def test_xlm_xlmmodel(self):
        """
        test xlm_TransformerFFN
        """
        xlm_xlmmodel = XLMModel(self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, self.config.vocab_size, (1, 1)), mindspore.int32)
        output = xlm_xlmmodel(input_ids=input_ids1)
        assert output[0].shape == (1, 1, 24)

    def test_xlm_xlmforsequenceclassification(self):
        """
        test xlm_XLMForSequenceClassification
        """
        xlm_xlmforsequenceclassification = XLMForSequenceClassification(
            self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 22, (1, 24)), mindspore.int32)
        output = xlm_xlmforsequenceclassification(input_ids=input_ids1)
        assert output[0].shape == (1, 2)

    def test_xlm_xlmforquestionansweringsimple(self):
        """
        test xlm_XLMForQuestionAnsweringSimple
        """

        xlm_xlmforquestionansweringsimple = XLMForQuestionAnsweringSimple(
            self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 22, (1, 24)), mindspore.int32)
        output = xlm_xlmforquestionansweringsimple(input_ids=input_ids1)
        assert output[0].shape == (1, 24)

    def test_xlm_xlmfortokenclassification(self):
        """
        test xlm_xlmfortokenclassification
        """
        xlm_xlmfortokenclassification = XLMForTokenClassification(
            self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 22, (1, 24)), mindspore.int32)
        output = xlm_xlmfortokenclassification(input_ids=input_ids1)
        assert output[0].shape == (1, 24, 2)

    def test_xlm_xlmformultiplechoice(self):
        """
        test xlm_xlmformultiplechoice
        """
        xlm_xlmformultiplechoice = XLMForMultipleChoice(self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 22, (22, 22)), mindspore.int32)
        output = xlm_xlmformultiplechoice(input_ids=input_ids1)
        assert output[0].shape == (1, 22)

    def test_xlm_xlmforquestionanswering(self):
        """
        test xlm_xlmforquestionanswering
        """
        xlm_xlmforquestionanswering = XLMForQuestionAnswering(self.config)
        input_ids1 = mindspore.Tensor(
            np.random.randint(0, 1, (13, 7)), mindspore.int32)
        # token_type_ids =  mindspore.Tensor(np.random.randint(0, 1, (13,7)),mindspore.int32)
        # input_lengths = mindspore.Tensor(np.random.randint(0, 1, (13)),mindspore.int32) + 7 - 2
        sequence_labels = mindspore.Tensor(
            np.random.randint(0, 1, (13)), mindspore.int32)
        # token_labels = mindspore.Tensor(np.random.randint(0, 1, (13,7)),mindspore.int32)
        is_impossible_labels = mindspore.Tensor(
            np.random.randint(0, 1, (13,)), mindspore.float32)
        # choice_labels = mindspore.Tensor(np.random.randint(0, 3, (13)),mindspore.int32)
        attn_mask = mindspore.Tensor(
            np.random.randint(0, 1, (13, 7)), mindspore.int32)
        attn_mask[:, -1] = 1
        input_mask = attn_mask

        output = xlm_xlmforquestionanswering(
            input_ids=input_ids1,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
            p_mask=input_mask,
        )
        assert output[0].shape == ()

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = XLMModel.from_pretrained('xlm-clm-enfr-1024')

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = XLMModel.from_pretrained('xlm-mlm-en-2048', from_pt=True)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
