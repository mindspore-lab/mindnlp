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
Test Nezha
"""
import gc
import os
import random
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.transformers.models.nezha.nezha import (NezhaConfig,
                                  NezhaRelativePositionsEncoding,
                                  NezhaEmbeddings,
                                  NezhaSelfAttention,
                                  NezhaSelfOutput,
                                  NezhaAttention,
                                  NezhaIntermediate,
                                  NezhaOutput,
                                  NezhaLayer,
                                  NezhaEncoder,
                                  NezhaPooler,
                                  NezhaPredictionHeadTransform,
                                  NezhaLMPredictionHead,
                                  NezhaOnlyMLMHead,
                                  NezhaOnlyNSPHead,
                                  NezhaPreTrainingHeads,
                                  NezhaModel,
                                  NezhaForPreTraining,
                                  NezhaForMaskedLM,
                                  NezhaForNextSentencePrediction,
                                  NezhaForSequenceClassification,
                                  NezhaForMultipleChoice,
                                  NezhaForTokenClassification,
                                  NezhaForQuestionAnswering)

from .....common import MindNLPTestCase


class TestNezhaBasicModule(MindNLPTestCase):
    r"""
    Test Nezha Basic Module
    """
    def setUp(self):
        self.config = NezhaConfig(num_hidden_layers=2, hidden_size=48, intermediate_size=192, vocab_size=64)

    def test_nezha_relative_positions_encoding(self):
        r"""
        Test NezhaRelativePositionsEncoding
        """
        length = 10
        depth = 20
        model = NezhaRelativePositionsEncoding(length, depth)
        inputs = random.randint(0, 10)
        outputs = model(inputs)
        assert outputs.shape == (inputs, inputs, 20)

    def test_nezha_embeddings(self):
        r"""
        Test NezhaEmbeddings
        """
        model = NezhaEmbeddings(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_self_attention(self):
        r"""
        Test NezhaSelfAttention
        """
        model = NezhaSelfAttention(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_self_output(self):
        r"""
        Test NezhaSelfOutput
        """
        model = NezhaSelfOutput(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs, inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_attention(self):
        r"""
        Test NezhaAttention
        """
        model = NezhaAttention(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_intermediate(self):
        r"""
        Test NezhaIntermediate
        """
        model = NezhaIntermediate(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 192)

    def test_nezha_output(self):
        r"""
        Test NezhaOutput
        """
        model = NezhaOutput(self.config)
        inputs1 = Tensor(np.random.randn(4, 16, 192), mindspore.float32)
        inputs2 = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs1, inputs2)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_layer(self):
        r"""
        Test NezhaLayer
        """
        model = NezhaLayer(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_encoder(self):
        r"""
        Test NezhaEncoder
        """
        model = NezhaEncoder(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_pooler(self):
        r"""
        Test NezhaPooler
        """
        model = NezhaPooler(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 48)

    def test_nezha_prediction_head_transform(self):
        r"""
        Test NezhaPredictionHeadTransform
        """
        model = NezhaPredictionHeadTransform(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_lm_prediction_head(self):
        r"""
        Test NezhaLMPredictionHead
        """
        model = NezhaLMPredictionHead(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 64)

    def test_nezha_only_mlm_head(self):
        r"""
        Test NezhaOnlyMLMHead
        """
        model = NezhaOnlyMLMHead(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 64)

    def test_nezha_only_nsp_head(self):
        r"""
        Test NezhaOnlyNSPHead
        """
        model = NezhaOnlyNSPHead(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 2)

    def test_nezha_pretraining_heads(self):
        r"""
        Test NezhaPreTrainingHeads
        """
        model = NezhaPreTrainingHeads(self.config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs, inputs)
        assert outputs[0].shape == (4, 16, 64)
        assert outputs[1].shape == (4, 16, 2)


class TestModelingNezha(MindNLPTestCase):
    r"""
    Test Nezha Model
    """
    def setUp(self):
        self.config = NezhaConfig(num_hidden_layers=2, hidden_size=48, intermediate_size=192, vocab_size=64)

    def test_nezha_model(self):
        r"""
        Test NezhaModel
        """
        model = NezhaModel(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)
        assert outputs[1].shape == (4, 48)

    def test_nezha_for_pretraining(self):
        r"""
        Test NezhaForPreTraining
        """
        model = NezhaForPreTraining(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 64)
        assert outputs[1].shape == (4, 2)

    def test_nezha_for_masked_lm(self):
        r"""
        Test NezhaForMaskedLM
        """
        model = NezhaForMaskedLM(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 64)

    def test_nezha_for_next_sentence_prediction(self):
        r"""
        Test NezhaForNextSentencePrediction
        """
        model = NezhaForNextSentencePrediction(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 2)

    def test_nezha_for_sequence_classification(self):
        r"""
        Test NezhaForSequenceClassification
        """
        model = NezhaForSequenceClassification(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 2)

    def test_nezha_for_multiple_choice(self):
        r"""
        Test NezhaForMultipleChoice
        """
        model = NezhaForMultipleChoice(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 4)

    def test_nezha_for_token_classification(self):
        r"""
        Test NezhaforTokenClassification
        """
        model = NezhaForTokenClassification(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 2)

    def test_for_question_answering(self):
        r"""
        Test NezhaForQuestionAnswering
        """
        model = NezhaForQuestionAnswering(self.config)
        inputs = Tensor(np.random.randint(0, self.config.vocab_size, (4, 16)), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16)
        assert outputs[1].shape == (4, 16)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
